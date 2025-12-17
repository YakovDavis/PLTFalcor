
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
    StructuredBuffer<BounceData> firstBounceBuffer;
    StructuredBuffer<hwss_t> gWavelengthsGI;

    void execute(const uint2 pixel)
    {
        if (any(pixel >= gFrameDim)) return;

        // Get index for the structured buffer access.
        uint bufferIndex = pixel.y * gFrameDim.x + pixel.x;

        // Create sample generator.
        SampleGenerator sg = SampleGenerator(pixel, gFrameCount);

        // Get reservoir with initial candidate samples from the previous pass.
        ReservoirGI currentReservoir = ReservoirGI::unpack(gGIReservoirs[bufferIndex]);

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
        // Use reprojPos directly if neighbor validation failed, otherwise use the validated prevPixel
        uint2 reusePixel = neighborFound ? prevPixel : reprojPos;
        uint prevBufferIndex = reusePixel.y * gFrameDim.x + reusePixel.x;

        // Clamp to valid bounds
        if (any(reusePixel >= gFrameDim) || any(reusePixel < 0)) {
            gPrevGIReservoirs[bufferIndex] = currentReservoir.pack();
            return;
        }

        ReservoirGI outputReservoir;
        // Get final reservoir from the previous frame.
        ReservoirGI prevReservoir = ReservoirGI::unpack(gPrevGIReservoirs[prevBufferIndex]);

        // Clamp the previous frame's M.
        prevReservoir.M = min(prevReservoir.M, kMCap);

        // Standard combination of multiple reservoirs (Algorithm 4 from the original paper):
        if (prevReservoir.M > 0)
        {
            //float3 dir = normalize(currentReservoir.sls.posW - v.posW);
            float prevPdf = prevReservoir.cachedIntensity; // * bsdf.pdf(sd, bounce.wi, gWavelengthsGI[bufferIndex][0]);
            outputReservoir.update(prevReservoir, prevPdf, sg);
        }

        if (currentReservoir.M > 0)
        {
            //float3 dir = normalize(currentReservoir.sls.posW - v.posW);
            float currPdf = currentReservoir.cachedIntensity; //  * bsdf.pdf(sd, bounce.wi, gWavelengthsGI[bufferIndex][0]);
            outputReservoir.update(currentReservoir, currPdf, sg);
        }

        outputReservoir.W = outputReservoir.W  > 0.f ? outputReservoir.weightSum / (outputReservoir.M * outputReservoir.W) : 0.f;

        // Store combined reservoir.
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
