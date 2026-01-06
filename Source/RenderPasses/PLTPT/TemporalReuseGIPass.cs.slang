
#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Path;
import Utils.Sampling.SampleGenerator;

import Utils.Color.ColorHelpers;

import ReservoirGI;
import SurfaceData;
import ReSTIRHelperFunctions;

import Scene.Raytracing;
import Scene.Intersection;
import Scene.ShadingData;
import Scene.Scene;

struct TemporalReuseGIPass
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
    RWStructuredBuffer<PackedReservoirGI> gPrevGIReservoirs;  ///< Buffer with reservoirs from the previous frame.
    StructuredBuffer<PackedReservoirGI> gGIReservoirs; ///< Buffer with reservoirs.
    StructuredBuffer<BounceData> firstBounceBuffer; ///< Buffer with first bounces from the camera rays.
    StructuredBuffer<hwss_t> gWavelengthsGI; ///< Wavelengths used in current pass.

    void execute(const uint2 pixel)
    {
        if (any(pixel >= gFrameDim)) return;

        uint bufferIndex = pixel.y * gFrameDim.x + pixel.x;

        SampleGenerator sg = SampleGenerator(pixel, gFrameCount);

        ReservoirGI currentReservoir = ReservoirGI::unpack(gGIReservoirs[bufferIndex]);

        // Reproject the pixel position
        float2 motionVec = gMotionVectors[pixel] * gFrameDim;
        float2 reprojPosFloat = float2(pixel) + motionVec;

        int2 reprojPosInt = int2(round(reprojPosFloat));

        if (any(reprojPosInt < int2(1)) || any(reprojPosInt >= int2(gFrameDim))) {
            gPrevGIReservoirs[bufferIndex] = currentReservoir.pack();
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

            if (!isValidNeighbor(surfaceData.normal, prevSurfaceData.normal, surfaceData.depth, prevSurfaceData.depth, kDepthThreshold, kNormalThreshold))
                continue;

            neighborFound = true;
            break;
        }

        if (!neighborFound) {
            gPrevGIReservoirs[bufferIndex] = currentReservoir.pack();
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

        ReservoirGI outputReservoir;

        ReservoirGI prevReservoir = ReservoirGI::unpack(gPrevGIReservoirs[prevBufferIndex]);

        prevReservoir.M = min(prevReservoir.M, kMCap);

        if (prevReservoir.M > 0)
        {
            float prevPdf = prevReservoir.cachedIntensity;
            outputReservoir.update(prevReservoir, prevPdf, sg);
        }

        if (currentReservoir.M > 0)
        {
            float currPdf = currentReservoir.cachedIntensity;
            outputReservoir.update(currentReservoir, currPdf, sg);
        }

        outputReservoir.W = outputReservoir.W  > 0.f ? outputReservoir.weightSum / (outputReservoir.M * outputReservoir.W) : 0.f;

        gPrevGIReservoirs[bufferIndex] = outputReservoir.pack();
    }
}

cbuffer CB
{
    TemporalReuseGIPass gTemporalReuseGIPass;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gTemporalReuseGIPass.execute(dispatchThreadId.xy);
}
