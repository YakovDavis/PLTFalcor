
#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Path;
import Utils.Sampling.SampleGenerator;

import Utils.Color.ColorHelpers;

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

struct TemporalReusePass
{
    static const uint kMCap = 5;
    static const float kMaxOffset = 5.f;
    static const uint kAttemptCount = 1;

    uint2   gFrameDim; ///< Frame dimensions.
    uint    gFrameCount; ///< Frame count since scene was loaded.

    // Resources:
    Texture2D<float2> gMotionVectors; ///< 2D screen space motion vector.
    StructuredBuffer<PackedSurfaceData> gSurfaceData;   ///< Surface data for the current frame.
    StructuredBuffer<PackedSurfaceData> gPrevSurfaceData;   ///< Surface data for the previous frame.
    RWStructuredBuffer<PackedReservoir> gPrevReservoirs;  ///< Buffer with reservoirs from the previous frame.
    StructuredBuffer<PackedReservoir> gReservoirs; ///< Buffer with reservoirs.
    StructuredBuffer<BounceData> firstBounceBuffer;
    StructuredBuffer<hwss_t> gWavelengthsDI;

    void execute(const uint2 pixel)
    {
        if (any(pixel >= gFrameDim)) return;

        // Get index for the structured buffer access.
        uint bufferIndex = pixel.y * gFrameDim.x + pixel.x;

        // Create sample generator.
        SampleGenerator sg = SampleGenerator(pixel, gFrameCount);

        // Get reservoir with initial candidate samples from the previous pass.
        Reservoir currentReservoir = Reservoir::unpack(gReservoirs[bufferIndex]);

        // Reproject the pixel position.
        uint2 reprojPos = uint2(float2(pixel) + gMotionVectors[pixel] * gFrameDim + sampleNext2D(sg));

        // Get surface data for the current pixel.
        SurfaceData surfaceData = SurfaceData::unpack(gSurfaceData[bufferIndex]);
        if (!surfaceData.isValid())
            return;

        SurfaceData prevSurfaceData;
        uint2 prevPixel = reprojPos;
        bool neighborFound = false;

        [unroll]
        for (uint i = 0; i < kAttemptCount; i++)
        {
            int2 offset = int2(0, 0);
            if (i > 0) {

                offset.x = int((sampleNext1D(sg) - 0.5) * kMaxOffset);
                offset.y = int((sampleNext1D(sg) - 0.5) * kMaxOffset);
            }
            prevPixel = reprojPos + offset;

            // Test if previous pixel position is out of bounds.
            if (any(prevPixel >= gFrameDim) || any(prevPixel < 1)) // 1 and not 0 because 0 causes artifacts
                continue;

            // Load shading data for the previous pixel.
            prevSurfaceData = SurfaceData::unpack(gPrevSurfaceData[getBufferIndex(prevPixel, gFrameDim)]);
            if (!prevSurfaceData.isValid())
                continue;

            // Compare the difference in camera distance, and the angle between normals.
            if (!isValidNeighbor(surfaceData.normal, prevSurfaceData.normal, surfaceData.depth, prevSurfaceData.depth, kDepthThreshold, kNormalThreshold))
                continue;

            neighborFound = true;
            break;
        }

        if (!neighborFound) {
            gPrevReservoirs[bufferIndex] = currentReservoir.pack();
            return;
        }

        // Retrieve bounce for bsdf eval
        const BounceData bounce = firstBounceBuffer[bufferIndex];

        const TriangleHit triangleHit = TriangleHit(bounce.hit);
        const VertexData v = gScene.getVertexData(triangleHit);

        let lod = ExplicitLodTextureSampler(0.f);
        const uint materialID = gScene.getMaterialID(triangleHit.instanceID);
        const ShadingData sd = gScene.materials.prepareShadingData(v, materialID, bounce.wi, lod);

        let bsdf = gScene.materials.getMaterialInstance(sd, lod);

        const uint lobes = bsdf.getLobes(sd);

        // Get index for the structured buffer access for the previous pixel.
        uint prevBufferIndex = prevPixel.y * gFrameDim.x + prevPixel.x;

        Reservoir outputReservoir = {};
        outputReservoir.hwss_comps = currentReservoir.hwss_comps;
        // Get final reservoir from the previous frame.
        Reservoir prevReservoir = Reservoir::unpack(gPrevReservoirs[prevBufferIndex]);

        // Clamp the previous frame's M.
        prevReservoir.M = min(prevReservoir.M, kMCap);

        // Standard combination of multiple reservoirs (Algorithm 4 from the original paper):
        if (prevReservoir.M > 0)
        {
            float3 dir = normalize(prevReservoir.sls.posW - v.posW);
            float prevPdf = evalTargetPDF(sd, bsdf, prevReservoir.cachedIntensity, lobes, prevReservoir.hwss_comps, gWavelengthsDI[bufferIndex], bounce.wi, dir);
            outputReservoir.update(prevReservoir, prevPdf, sg);
        }

        if (currentReservoir.M > 0)
        {
            float3 dir = normalize(currentReservoir.sls.posW - v.posW);
            float currPdf = evalTargetPDF(sd, bsdf, currentReservoir.cachedIntensity, lobes, currentReservoir.hwss_comps, gWavelengthsDI[bufferIndex], bounce.wi, dir);
            outputReservoir.update(currentReservoir, currPdf, sg);
        }

        outputReservoir.W = outputReservoir.W  > 0.f ? (outputReservoir.weightSum / outputReservoir.M) / outputReservoir.W : 0.f;

        // Store combined reservoir.
        gPrevReservoirs[bufferIndex] = outputReservoir.pack();
    }
}

cbuffer CB
{
    TemporalReusePass gTemporalReusePass;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gTemporalReusePass.execute(dispatchThreadId.xy);
}
