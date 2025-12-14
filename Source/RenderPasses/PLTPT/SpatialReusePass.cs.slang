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

struct SpatialReusePass
{
    static const float FLT_LARGE = 1e20f;
    static const uint kNeighborCount = 5;
    static const float kDepthThreshold = 0.1f;
    static const float kNormalThreshold = 0.2f;

    uint2   gFrameDim; ///< Frame dimensions.
    uint    gFrameCount; ///< Frame count since scene was loaded.

    // Resources
    StructuredBuffer<PackedReservoir> gReservoirs; ///< Buffer with packed reservoirs.
    StructuredBuffer<PackedPLTBeam> beamBuffer; ///< Buffer with packed PLT beams.
    StructuredBuffer<PackedSurfaceData> gSurfaceData;     ///< Surface data for the current frame.
    RWStructuredBuffer<PackedReservoir> gOutReservoirs; ///< Output buffer with packed reservoirs.
    RWStructuredBuffer<PackedPLTBeam> outBeamBuffer; ///< Output buffer with packed PLT beams.
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

        // Create output reservoir.
        Reservoir outputReservoir;

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
            float3 dir = normalize(currentReservoir.sls.posW - v.posW);
            float currPdf = evalTargetPDF(sd, bsdf, currentReservoir.cachedIntensity, lobes, currentReservoir.hwss_comps, gWavelengthsDI[bufferIndex], bounce.wi, dir);
            outputReservoir.update(currentReservoir, currPdf, sg);
        }

        uint validNeighborFlags = 0;

#if true
        SampleGenerator sg1 = sg;
        float radius = 30.0f;
        [unroll]
        for (int i = 0; i < kNeighborCount; i++)
        {
            uint2 neighborPixel = getRandomNeighborPixel(pixel, sg1, radius);

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

            if (!isValidNeighbor(surfaceData.normal, neighborSurfaceData.normal, surfaceData.depth, neighborSurfaceData.depth, kDepthThreshold, kNormalThreshold))
            {
                continue;
            }

            if (!evalSampleVisibility(v.posW, neighborReservoir.sls.posW))
            {
                continue;
            }

            if (neighborReservoir.M > 0)
            {
                float3 dir = normalize(neighborReservoir.sls.posW - v.posW);
                float neighborPdf = evalTargetPDF(sd, bsdf, neighborReservoir.cachedIntensity, lobes, neighborReservoir.hwss_comps, gWavelengthsDI[bufferIndex], bounce.wi, dir);
                outputReservoir.update(neighborReservoir, neighborPdf, sg);
            }
        }
        outputReservoir.W = outputReservoir.W > 0.f ? (outputReservoir.weightSum / outputReservoir.M) / (outputReservoir.W) : 0.f;
#endif

	    // Store reservoir
        gOutReservoirs[bufferIndex] = outputReservoir.pack();
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
