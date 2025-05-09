
#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

//import Scene.Raytracing;
//import Utils.Geometry.GeometryHelpers;

//__exported import Scene.Shading;
//__exported import Utils.Timing.GpuTimer;
//__exported import Utils.Math.Ray;
//import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;

//import Scene.SceneTypes;
//import Scene.ShadingData;
//import Utils.Math.MathHelpers;
//import Utils.Math.FormatConversion;
//__exported import Scene.Material.ShadingUtils;

import Utils.Color.ColorHelpers;

import Reservoir;
import SurfaceData;
import ReSTIRHelperFunctions;

struct TemporalReusePass
{
    static const uint kMCap = 30;
    static const float kMaxOffset = 5.f;
    static const uint kAttemptCount = 1;

    uint2   gFrameDim; ///< Frame dimensions.
    uint    gFrameCount; ///< Frame count since scene was loaded.

    // Resources:
    Texture2D<float2> gMotionVectors; ///< 2D screen space motion vector.
    StructuredBuffer<PackedSurfaceData> gSurfaceData;   ///< Surface data for the current frame.
    StructuredBuffer<PackedSurfaceData> gPrevSurfaceData;   ///< Surface data for the previous frame.
    StructuredBuffer<PackedReservoir> gPrevReservoirs; ///< Buffer with reservoirs from the previous frame.
    RWStructuredBuffer<PackedReservoir> gReservoirs; ///< Buffer with reservoirs.
    StructuredBuffer<PackedPLTBeam> prevBeamBuffer; ///< Buffer with reservoir beams from the previous frame.
    RWStructuredBuffer<PackedPLTBeam> beamBuffer; ///< Buffer with reservoir beams

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
            if (any(prevPixel >= gFrameDim) || any(prevPixel < 0))
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
            return;
            //gReservoirs[bufferIndex] = currentReservoir.pack();
        }

        // Get index for the structured buffer access for the previous pixel.
        uint prevBufferIndex = prevPixel.y * gFrameDim.x + prevPixel.x;

        Reservoir outputReservoir;
        // Get final reservoir from the previous frame.
        Reservoir prevReservoir = Reservoir::unpack(gPrevReservoirs[prevBufferIndex]);

        // Clamp the previous frame's M.
        prevReservoir.M = min(prevReservoir.M, kMCap);

        // Standard combination of multiple reservoirs (Algorithm 4 from the original paper):
        float prevTargetPdf = 0.f;
        const uint prevBeam0Idx = (prevPixel.x + (prevPixel.y * gFrameDim.x)) * 4; // TODO: kMaxBeamCount;
        for (int p = 0; p < prevReservoir.beamCount; ++p) {
            PLTBeam b = PLTBeam::unpack(prevBeamBuffer[prevBeam0Idx + p]);
            prevTargetPdf += b.getRestirPdf();
        }
        bool prevChosen = outputReservoir.update(prevReservoir, prevTargetPdf, sg);

        float targetPdf = 0.f;
        const uint beam0Idx = (pixel.x + (pixel.y * gFrameDim.x)) * 4; // TODO: kMaxBeamCount;
        for (int p = 0; p < currentReservoir.beamCount; ++p) {
            PLTBeam b = PLTBeam::unpack(beamBuffer[beam0Idx + p]);
            targetPdf += b.getRestirPdf();
        }
        bool currentChosen = outputReservoir.update(currentReservoir, targetPdf, sg);

        outputReservoir.W = outputReservoir.W  > 0.f ? outputReservoir.weightSum / (outputReservoir.M * outputReservoir.W) : 0.f;

        // Store combined reservoir.
        gReservoirs[bufferIndex] = outputReservoir.pack();

        // Move beams from prev to current if prev chosen
        if (!currentChosen && prevChosen)
        {
            for (int p = 0; p < prevReservoir.beamCount; ++p) {
                beamBuffer[beam0Idx + p] = prevBeamBuffer[prevBeam0Idx + p];
            }
        }
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
