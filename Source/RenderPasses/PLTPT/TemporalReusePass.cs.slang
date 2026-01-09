
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

struct TemporalReusePass
{
    static const uint kMCap = 5;
    static const float kMaxOffset = 3.f;
    static const uint kAttemptCount = 1;
    static const float kDepthThreshold = 0.3f;
    static const float kNormalThreshold = -0.1f;

    uint2   gFrameDim; ///< Frame dimensions.
    uint    gFrameCount; ///< Frame count since scene was loaded.

    // Resources:
    Texture2D<float2> gMotionVectors; ///< 2D screen space motion vector.
    StructuredBuffer<PackedSurfaceData> gSurfaceData;   ///< Surface data for the current frame.
    StructuredBuffer<PackedSurfaceData> gPrevSurfaceData;   ///< Surface data for the previous frame.
    RWStructuredBuffer<PackedReservoir> gPrevReservoirs;  ///< Buffer with reservoirs from the previous frame.
    StructuredBuffer<PackedReservoir> gReservoirs; ///< Buffer with reservoirs.
    StructuredBuffer<BounceData> firstBounceBuffer; ///< Buffer with first bounces from the camera rays.
    StructuredBuffer<hwss_t> gWavelengthsDI; ///< Wavelengths used in current pass.

    void execute(const uint2 pixel)
    {
        if (any(pixel >= gFrameDim)) return;

        uint bufferIndex = pixel.y * gFrameDim.x + pixel.x;

        SampleGenerator sg = SampleGenerator(pixel, gFrameCount);

        Reservoir currentReservoir = Reservoir::unpack(gReservoirs[bufferIndex]);

        // Reproject the pixel position
        float2 motionVec = gMotionVectors[pixel] * gFrameDim;
        float2 reprojPosFloat = float2(pixel) + motionVec;

        int2 reprojPosInt = int2(round(reprojPosFloat));

        if (any(reprojPosInt < int2(1)) || any(reprojPosInt >= int2(gFrameDim))) {
            gPrevReservoirs[bufferIndex] = currentReservoir.pack();
            return;
        }
        uint2 reprojPos = uint2(reprojPosInt);

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
            int2 prevPixelInt = int2(reprojPos) + offset;
            if (any(prevPixelInt < int2(1)) || any(prevPixelInt >= int2(gFrameDim))) // 1 and not 0 because 0 causes artifacts
                continue;
            prevPixel = uint2(prevPixelInt);

            prevSurfaceData = SurfaceData::unpack(gPrevSurfaceData[getBufferIndex(prevPixel, gFrameDim)]);
            if (!prevSurfaceData.isValid())
                continue;

            if (!isValidNeighbor(surfaceData.normal, prevSurfaceData.normal, surfaceData.depth, prevSurfaceData.depth, kNormalThreshold, kDepthThreshold))
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

        const uint lobes = bounce.sampledBsdfLobe;

        uint prevBufferIndex = prevPixel.y * gFrameDim.x + prevPixel.x;

        // Clamp to valid bounds
        if (any(prevPixel >= gFrameDim) || any(prevPixel < 0)) {
            gPrevReservoirs[bufferIndex] = currentReservoir.pack();
            return;
        }

        Reservoir outputReservoir = {};
        outputReservoir.hwss_comps = currentReservoir.hwss_comps;

        Reservoir prevReservoir = Reservoir::unpack(gPrevReservoirs[prevBufferIndex]);

        prevReservoir.M = min(prevReservoir.M, kMCap);

        float3 lightPos = prevReservoir.sls.distant > 0 ? surfaceData.position + prevReservoir.sls.posW : prevReservoir.sls.posW;

        if (prevReservoir.M > 0)
        {
            float3 dir = normalize(lightPos - surfaceData.position);
            float prevPdf = prevReservoir.cachedIntensity;
            outputReservoir.update(prevReservoir, prevPdf, sg);
        }

        if (currentReservoir.M > 0)
        {
            float3 currLightPos = currentReservoir.sls.distant > 0 ? surfaceData.position + currentReservoir.sls.posW : currentReservoir.sls.posW;
            float3 dir = normalize(currLightPos - surfaceData.position);
            float currPdf = currentReservoir.cachedIntensity;
            outputReservoir.update(currentReservoir, currPdf, sg);
        }

        outputReservoir.W = outputReservoir.W  > 0.f ? (outputReservoir.weightSum / outputReservoir.M) / outputReservoir.W : 0.f;

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
