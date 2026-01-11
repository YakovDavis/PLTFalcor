#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Path;
import Utils.Geometry.GeometryHelpers;
import Scene.RaytracingInline;
import Utils.Sampling.SampleGenerator;

import Reservoir;
import SurfaceData;
import ReSTIRHelperFunctions;

import Scene.Raytracing;
import Scene.Intersection;
import Scene.ShadingData;
import Scene.Scene;

// Shadow ray epsilon is a small value used to nudge the shadow ray origin along the normal
// to avoid self-intersection due to numerical precision issues.
static const float kShadowRayEpsilon = 0.0001f;

struct SpatialReusePass
{
    static const uint kMCap = 30;
    static const float FLT_LARGE = 1e20f;
    static const uint kNeighborCount = 3;
    static const float kDepthThreshold = 0.3f;
    static const float kNormalThreshold = 0.9f;
    static const float kSearchRadius = 30.0f;

    uint2   gFrameDim; ///< Frame dimensions.
    uint    gFrameCount; ///< Frame count since scene was loaded.

    // Resources
    StructuredBuffer<PackedReservoir> gReservoirs; ///< Buffer with packed reservoirs.
    StructuredBuffer<PackedSurfaceData> gSurfaceData;     ///< Surface data for the current frame.
    RWStructuredBuffer<PackedReservoir> gOutReservoirs; ///< Output buffer with packed reservoirs.
    StructuredBuffer<BounceData> firstBounceBuffer;
    StructuredBuffer<hwss_t> gWavelengthsDI;

    bool evalSampleVisibility(float3 pos, float3 samplePoint)
    {
        float3 posToSample = samplePoint - pos;
        float distance = length(posToSample);
        float3 dir = posToSample / distance;
        const Ray ray = Ray(pos, dir, kShadowRayEpsilon, distance - kShadowRayEpsilon);
        SceneRayQuery<1> sceneRayQuery;
        return sceneRayQuery.traceVisibilityRay(ray, RAY_FLAG_NONE, 0xff);
    }

    void execute(const uint2 pixel)
    {
        if (any(pixel >= gFrameDim)) return;

		// Get index for the structured buffer access.
		uint bufferIndex = pixel.y * gFrameDim.x + pixel.x;

		// Create sample generator.
        SampleGenerator sg = SampleGenerator(pixel, gFrameCount);

        // Get reservoir with initial candidate samples from the previous pass.
        Reservoir currentReservoir = Reservoir::unpack(gReservoirs[bufferIndex]);

        Reservoir outputReservoir = {};
        outputReservoir.hwss_comps = currentReservoir.hwss_comps;

        // Get pixel Data for the current pixel.
        SurfaceData surfaceData = SurfaceData::unpack(gSurfaceData[bufferIndex]);
        if (!surfaceData.isValid())
            return;

        // Retrieve bounce for bsdf eval
        const BounceData bounce = firstBounceBuffer[bufferIndex];

        const TriangleHit triangleHit = TriangleHit(bounce.hit);
        const VertexData v = gScene.getVertexData(triangleHit);

        let lod = ExplicitLodTextureSampler(0.f);
        const uint materialID = gScene.getMaterialID(triangleHit.instanceID);
        const ShadingData sd = gScene.materials.prepareShadingData(v, materialID, bounce.wi, lod);

        let bsdf = gScene.materials.getMaterialInstance(sd, lod);

        const uint lobes = bounce.sampledBsdfLobe;

        if (currentReservoir.M > 0)
        {
            float3 lightPos = currentReservoir.sls.distant > 0 ? surfaceData.position + currentReservoir.sls.posW : currentReservoir.sls.posW;
            float3 dir = normalize(lightPos - surfaceData.position);
            float currPdf = evalTargetPDF(sd, bsdf, currentReservoir, lobes, gWavelengthsDI[bufferIndex], bounce.wi, dir);;
            outputReservoir.update(currentReservoir, currPdf, sg);
        }

        float3 rayOrigin = computeRayOrigin(surfaceData.position, surfaceData.normal);

        float pSum = currentReservoir.cachedIntensity * currentReservoir.M;
		float pStar = currentReservoir.cachedIntensity;
        [unroll]
        for (int i = 0; i < kNeighborCount; i++)
        {
            uint2 neighborPixel = getRandomNeighborPixel(pixel, sg, kSearchRadius);

            // Discard pixel if out of bounds
	        if(any(neighborPixel >= gFrameDim) || any(neighborPixel < 0))
                continue;

            uint neighborBufferIndex = getBufferIndex(neighborPixel, gFrameDim);
            Reservoir neighborReservoir = Reservoir::unpack(gReservoirs[neighborBufferIndex]);

            SurfaceData neighborSurfaceData = SurfaceData::unpack(gSurfaceData[neighborBufferIndex]);
            if (!neighborSurfaceData.isValid())
            {
                continue;
            }

            if (neighborReservoir.M == 0.f)
            {
                continue;
            }

            if (!isValidNeighbor(surfaceData.normal, neighborSurfaceData.normal, surfaceData.depth, neighborSurfaceData.depth, kNormalThreshold, kDepthThreshold))
            {
                continue;
            }

            float3 lightPos = neighborReservoir.sls.distant > 0 ? surfaceData.position + neighborReservoir.sls.posW : neighborReservoir.sls.posW;

            if (!evalSampleVisibility(rayOrigin, lightPos))
            {
                continue;
            }

            float3 dir = lightPos - surfaceData.position;
            float distSq = dot(dir, dir);
            if (distSq < FLT_EPSILON || !all(isfinite(dir)))
            {
                continue;
            }
            dir = normalize(dir);

            // Check if direction is valid for the current surface (not pointing through backface unless transmission)
            const bool isTransmission = (lobes & (uint)LobeType::NonDeltaTransmission) != 0 || (lobes & (uint)LobeType::DeltaTransmission) != 0;
            const float cosTheta = dot(surfaceData.normal, dir);
            if (!isTransmission && cosTheta <= 0.f)
            {
                continue;
            }

            float neighborPixelTargetPDF = evalTargetPDF(sd, bsdf, neighborReservoir, lobes, gWavelengthsDI[bufferIndex], bounce.wi, dir);
            if (isfinite(neighborPixelTargetPDF) && neighborPixelTargetPDF > 0)
            {
                pSum += neighborPixelTargetPDF * neighborReservoir.M;
                if (outputReservoir.update(neighborReservoir, neighborPixelTargetPDF, sg)) {
                    pStar = neighborPixelTargetPDF;
                }
            }
        }

        if (outputReservoir.M > 0)
        {
            //outputReservoir.W = outputReservoir.W > 0.f ? (outputReservoir.weightSum / outputReservoir.M) / (outputReservoir.W) : 0.f;
            float m = pSum > 0.f ? pStar / pSum : 0.f;
            outputReservoir.W = outputReservoir.W > 0.f ? (outputReservoir.weightSum * m) / outputReservoir.W : 0.f;
            outputReservoir.M = min(outputReservoir.M, kMCap);
            gOutReservoirs[bufferIndex] = outputReservoir.pack();
        }
        else
        {
            gOutReservoirs[bufferIndex] = currentReservoir.pack();
        }
    }
}

cbuffer CB
{
    SpatialReusePass gSpatialReusePass;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gSpatialReusePass.execute(dispatchThreadId.xy);
}
