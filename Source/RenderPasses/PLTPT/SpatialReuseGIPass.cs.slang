#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Path;
import Utils.Geometry.GeometryHelpers;
import Scene.RaytracingInline;
import Utils.Sampling.SampleGenerator;

import ReservoirGI;
import SurfaceData;
import ReSTIRHelperFunctions;

import Scene.Raytracing;
import Scene.Intersection;
import Scene.ShadingData;
import Scene.Scene;

float evalTargetPDF(const ShadingData sd, const IMaterialInstance bsdf, const float cachedIntensity, const uint lobe, const uint hwss_comps, hwss_t wavelengths, const float3 wi, const float3 wo)
{
    PLTBeam testBeam = {};
    testBeam.wavelengths = wavelengths;
    testBeam.r = wo;
    testBeam.hwss_comps = hwss_comps;
    testBeam.L = hwss_t(cachedIntensity);
    bsdf.eval(sd, wi, lobe, testBeam);
    float intensitySum = 0;
    for (int i = 0; i < hwss_comps; ++i)
    {
        intensitySum += testBeam.L[i];
    }
    return intensitySum / hwss_comps;
}

// Shadow ray epsilon is a small value used to nudge the shadow ray origin along the normal
// to avoid self-intersection due to numerical precision issues.
static const float kShadowRayEpsilon = 0.0001f;

struct SpatialReuseGIPass
{
    static const float FLT_LARGE = 1e20f;
    static const uint kMCap = 300;
    static const uint kNeighborCount = 5;
    static const float kDepthThreshold = 0.2f;
    static const float kNormalThreshold = 0.5f;
    static const float kMinRadius = 100.f;

    uint2   gFrameDim; ///< Frame dimensions.
    uint    gFrameCount; ///< Frame count since scene was loaded.

    // Resources
    StructuredBuffer<PackedReservoirGI> gGIReservoirs; ///< Buffer with packed reservoirs.
    StructuredBuffer<PackedSurfaceData> gSurfaceData;     ///< Surface data for the current frame.
    RWStructuredBuffer<PackedReservoirGI> gOutGIReservoirs; ///< Output buffer with packed reservoirs.
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
        ReservoirGI currentReservoir = ReservoirGI::unpack(gGIReservoirs[bufferIndex]);

        // Create output reservoir.
        ReservoirGI outputReservoir;

        const float3 primaryRayOrigin = gScene.camera.getPosition();
        const float3 primaryRayDir = getPrimaryRayDir(pixel, gFrameDim, gScene.camera);
        const float3 viewVec = -normalize(primaryRayDir);

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

        const uint lobes = bsdf.getLobes(sd);

        if (currentReservoir.M > 0)
        {
            const TriangleHit triangleHit = TriangleHit(currentReservoir.bounces[0].hit);
            const VertexData v1 = gScene.getVertexData(triangleHit);
            float3 dir = normalize(v1.posW - v.posW);
            float currPdf = evalTargetPDF(sd, bsdf, currentReservoir.cachedIntensity, lobes, currentReservoir.hwss_comps, gWavelengthsDI[bufferIndex], bounce.wi, dir);
            outputReservoir.update(currentReservoir, currPdf, sg);
        }

        float3 rayOrigin = computeRayOrigin(surfaceData.position, surfaceData.normal);
        SampleGenerator sg1 = sg;

        float radius = 0.05f * gFrameDim.y;
        [unroll]
        for (int i = 0; i < kNeighborCount; i++)
        {
            uint2 neighborPixel = getRandomNeighborPixel(pixel, sg1, radius);

            // Discard pixel if out of bounds
	        if(any(neighborPixel >= gFrameDim) || any(neighborPixel < 0))
                continue;

            uint neighborBufferIndex = getBufferIndex(neighborPixel, gFrameDim);
            ReservoirGI neighborReservoir = ReservoirGI::unpack(gGIReservoirs[neighborBufferIndex]);

            SurfaceData neighborSurfaceData = SurfaceData::unpack(gSurfaceData[neighborBufferIndex]);
            if (!neighborSurfaceData.isValid())
            {
                radius = max(kMinRadius, radius * 0.5f);
                continue;
            }

            if (neighborReservoir.M == 0.f)
            {
                radius = max(kMinRadius, radius * 0.5f);
                continue;
            }

            if (!isValidNeighbor(surfaceData.normal, neighborSurfaceData.normal, surfaceData.depth, neighborSurfaceData.depth, kDepthThreshold, kNormalThreshold))
            {
                radius = max(kMinRadius, radius * 0.5f);
                continue;
            }

            float neighborReservoirTargetPDF = 0.f;
            float3 neighborSamplePoint = float3(0.f);
            float3 neighborSampleNormal = float3(0.f);
            {
                const TriangleHit triangleHit = TriangleHit(neighborReservoir.bounces[0].hit);
                const VertexData v1 = gScene.getVertexData(triangleHit);
                neighborSamplePoint = v1.posW;
                neighborSampleNormal = v1.normalW;
                float3 dir = normalize(v1.posW - v.posW);
                neighborReservoirTargetPDF = evalTargetPDF(sd, bsdf, neighborReservoir.cachedIntensity, lobes, neighborReservoir.hwss_comps, gWavelengthsDI[bufferIndex], bounce.wi, dir);
            }

            if (any(neighborSampleNormal != 0.f)) // otherwise only direct illumination, jacobian = 1
            {
                float3 r1q2 = neighborSamplePoint - surfaceData.position;
                float3 q1q2 = neighborSamplePoint - neighborSurfaceData.position;

                // Discard back-face samples
                if (dot(surfaceData.normal, r1q2) <= 0.f)
                    neighborReservoirTargetPDF = 0.f;

                float r1q2LenSq = dot(r1q2, r1q2);
                float q1q2LenSq = dot(q1q2, q1q2);

                r1q2 = normalize(r1q2);
                q1q2 = normalize(q1q2);

                float cosA = dot(surfaceData.normal, r1q2);
                float cosB = dot(neighborSampleNormal, q1q2);
                float cosPhiA = -dot(r1q2, neighborSampleNormal);
                float cosPhiB = -dot(q1q2, neighborSampleNormal);
                if (cosB <= 0.f || cosPhiB <= 0.f)
                {
                    radius = max(kMinRadius, radius * 0.5f);
                    continue;
                }

                if (cosA <= 0.f || cosPhiA <= 0.f || r1q2LenSq <= 0.f || q1q2LenSq <= 0.f)
                {
                    neighborReservoirTargetPDF = 0.f;
                }

                // Evaluate visibility
                bool isVisible = evalSampleVisibility(rayOrigin, neighborSamplePoint);
                if (!isVisible)
                    neighborReservoirTargetPDF = 0.f;

                // Calculate Jacobian determinant
                float jacobian = r1q2LenSq * cosPhiB <= 0.f ? 0.f : clamp(q1q2LenSq * cosPhiA / (r1q2LenSq * cosPhiB), 0.f, FLT_LARGE);
                neighborReservoirTargetPDF *= jacobian;
            }

            bool neighborChosen = outputReservoir.update(neighborReservoir, neighborReservoirTargetPDF, sg);
        }

        outputReservoir.W = outputReservoir.W > 0.f ? (outputReservoir.weightSum / outputReservoir.M) / (outputReservoir.W) : 0.f;
        outputReservoir.M = min(outputReservoir.M, kMCap);

	    // Store reservoir
        gOutGIReservoirs[bufferIndex] = outputReservoir.pack();
    }
}

cbuffer CB
{
    SpatialReuseGIPass gSpatialReuseGIPass;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gSpatialReuseGIPass.execute(dispatchThreadId.xy);
}
