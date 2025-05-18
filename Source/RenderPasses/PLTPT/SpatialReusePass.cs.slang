#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Utils.Geometry.GeometryHelpers;
import Scene.RaytracingInline;
import Utils.Sampling.SampleGenerator;

import Reservoir;
import SurfaceData;
import ReSTIRHelperFunctions;

// Shadow ray epsilon is a small value used to nudge the shadow ray origin along the normal
// to avoid self-intersection due to numerical precision issues.
static const float kShadowRayEpsilon = 0.0001f;

struct SpatialReusePass
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
    StructuredBuffer<PackedReservoir> gReservoirs; ///< Buffer with packed reservoirs.
    StructuredBuffer<PackedPLTBeam> beamBuffer; ///< Buffer with packed PLT beams.
    StructuredBuffer<PackedSurfaceData> gSurfaceData; ///< Surface data for the current frame.
    RWStructuredBuffer<PackedReservoir> gOutReservoirs; ///< Output buffer with packed reservoirs.
    RWStructuredBuffer<PackedPLTBeam> outBeamBuffer; ///< Output buffer with packed PLT beams.

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

        float currReservoirTargetPDF = 0.f;
        const uint currBeam0Idx = (pixel.x + (pixel.y * gFrameDim.x)) * 4; // TODO: kMaxBeamCount;
        for (int p = 0; p < currentReservoir.beamCount; ++p) {
            PLTBeam b = PLTBeam::unpack(beamBuffer[currBeam0Idx + p]);
            currReservoirTargetPDF += b.getRestirPdf();
        }
        bool currChosen = outputReservoir.update(currentReservoir, currReservoirTargetPDF, sg);
        // Move beams from current to output if chosen
        if (currChosen)
        {
            for (int p = 0; p < currentReservoir.beamCount; ++p) {
                outBeamBuffer[currBeam0Idx + p] = beamBuffer[currBeam0Idx + p];
            }
        }

        float3 rayOrigin = computeRayOrigin(surfaceData.position, surfaceData.normal);
        SampleGenerator sg1 = sg;

        float radius = 0.1f * gFrameDim.y;
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
            const uint neighborBeam0Idx = (pixel.x + (pixel.y * gFrameDim.x)) * 4; // TODO: kMaxBeamCount;
            float3 neighborSamplePoint = float3(0.f);
            for (int p = 0; p < currentReservoir.beamCount; ++p) {
                PLTBeam b = PLTBeam::unpack(beamBuffer[neighborBeam0Idx + p]);
                if (p == 0) // Record a (shared) beam position to check visibility
                {
                    neighborSamplePoint = b.posW;
                }
                neighborReservoirTargetPDF += b.getRestirPdf();
            }

            float3 r1q2 = neighborSamplePoint - surfaceData.position;
            float3 q1q2 = neighborSamplePoint - neighborSurfaceData.position;

            // Discard back-face samples
            if (dot(surfaceData.normal, r1q2) <= 0.f)
                neighborReservoirTargetPDF = 0.f;

            float r1q2LenSq = dot(r1q2, r1q2);
            float q1q2LenSq = dot(q1q2, q1q2);

            r1q2 = normalize(r1q2);
            q1q2 = normalize(q1q2);

            /*float cosA = dot(surfaceData.normal, r1q2);
            float cosB = dot(neighborLightSample.surfaceNormal, q1q2);
            float cosPhiA = -dot(r1q2, neighborLightSample.sampleNormal);
            float cosPhiB = -dot(q1q2, neighborLightSample.sampleNormal);
            if (cosB <= 0.f || cosPhiB <= 0.f) {
                radius = max(kMinRadius, radius * 0.5f);
                continue;
            }

            if (cosA <= 0.f || cosPhiA <= 0.f || r1q2LenSq <= 0.f || q1q2LenSq <= 0.f) {

                neighborReservoirTargetPDF = 0.f;
            }*/

            // Evaluate visibility
            bool isVisible = evalSampleVisibility(rayOrigin, neighborSamplePoint);
            if (!isVisible)
                neighborReservoirTargetPDF = 0.f;

            // Calculate Jacobian determinant
            //float jacobian = r1q2LenSq * cosPhiB <= 0.f ? 0.f : clamp(q1q2LenSq * cosPhiA / (r1q2LenSq * cosPhiB), 0.f, FLT_LARGE);
            //neighborReservoirTargetPDF *= jacobian;

            bool neighborChosen = outputReservoir.update(neighborReservoir, neighborReservoirTargetPDF, sg);
            // Move beams from neighbor to output if neighbor chosen
            if (neighborChosen)
            {
                for (int p = 0; p < neighborReservoir.beamCount; ++p) {
                    outBeamBuffer[currBeam0Idx + p] = beamBuffer[neighborBeam0Idx + p];
                }
            }
        }

        outputReservoir.W = outputReservoir.W > 0.f ? (outputReservoir.weightSum / outputReservoir.M) / (outputReservoir.W) : 0.f;
        outputReservoir.M = min(outputReservoir.M, kMCap);

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
