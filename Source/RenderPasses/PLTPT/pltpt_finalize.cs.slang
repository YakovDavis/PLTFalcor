/***************************************************************************
 # PLT
 # Copyright (c) 2022-23, Shlomi Steinberg. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/

#include "pltpt.slangh"

#include "LightSampling.slangh"

StructuredBuffer<BounceData> firstBounceBuffer;
StructuredBuffer<hwss_t> gWavelengthsDI;
StructuredBuffer<hwss_t> gWavelengthsGI;

/** Path measurement */

void forwardHWSSMis(const IMaterialInstance bsdf,
                    const ShadingData sd,
                    const float3 wo,
                    const uint sampledBsdfLobe,
                    inout PLTBeam beam) {
    if (mHWSSDoMIS) {
        const hwss_t pdfs = bsdf.lambdaPdf(sd, wo, sampledBsdfLobe, beam, beam.wavelengths);
        beam.spectralPDFs *= pdfs;
    }
}

// Measure beam at sensor
float3 measure(float pathContribution, inout PLTBeam beam) {

    const float k = M_2PI / (1e-3f * beam.wavelengths.x);

    // Propagate to camera
    beam.propagate(gScene.camera.data.posW);

    // Camera polarizer
    const float polarizationIntensity = gScene.camera.data.polarizer;
    const float polarizationTheta = gScene.camera.data.polarizerTheta;
    if (polarizationIntensity>.0f) {
        // Rotate to camera frame
        const float3 cameraW = normalize(gScene.camera.data.cameraW);
        const float3 cameraU = normalize(gScene.camera.data.cameraU);
        beam.rotateFrame(cameraU,-cameraW);

        // Polarize
        const float4x4 P = (1.f-polarizationIntensity) * ident4() +
                            polarizationIntensity * MuellerPolarizer(polarizationTheta);
        for (int j = 0; j < beam.hwss_comps; ++j)
            beam.setSP(j, mul(P, beam.getSP(j)));
    }

    // TODO: Camera/lens effects

    if (mHWSSDoMIS) {
        // MIS spectral contributions
        const float sum = dot(beam.spectralPDFs, hwss_t(1.f));
        pathContribution *= sum>FLT_EPSILON ? beam.spectralPDFs[0]/sum : .0f;
    }
    else
        pathContribution *= 1.f / float(beam.hwss_comps);

    const hwss_t Li = beam.L;
    const float3 Lrgb = max(float3(.0f), SpectrumUtils.spectrumToRgb(Li, beam.wavelengths, beam.hwss_comps) * pathContribution);

    if (all(isfinite(Lrgb)))
        return Lrgb;
    return float3(0.f);
}

void forwardUnwindStep(const TriangleHit triangleHit,
                       const VertexData v,
                       const float3 wo,
                       const uint sampledBsdfLobe,
                       inout PLTBeam beam) {
    let lod = ExplicitLodTextureSampler(0.f);
    const uint materialID = gScene.getMaterialID(triangleHit.instanceID);
    const ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -beam.r, lod);

    beam.lastHitNormal = sd.N;

    // Propagate
    beam.propagate(sd.posW);

    // BSDF
    let bsdf = gScene.materials.getMaterialInstance(sd, lod);
    bsdf.eval(sd, wo, sampledBsdfLobe, beam);

    // Spectral MIS
    forwardHWSSMis(bsdf, sd, wo, sampledBsdfLobe, beam);
}

// Replay path and evaluate contributions
void forwardUnwindDI(const BounceData bounce, const TriangleHit triangleHit, const VertexData v, const uint2 pixel, inout PLTBeam beam)
{
    const float3 wo = bounce.wi;
    forwardUnwindStep(triangleHit, v, wo, bounce.sampledBsdfLobe, beam);
}

// Replay path and evaluate contributions GI
void forwardUnwindGI(const BounceData bounce, const TriangleHit triangleHit, const VertexData v, const uint2 pixel, const ReservoirGI resGI, inout PLTBeam beam)
{
    // First we replay reservoir bounces
    for (int p=resGI.bounceCount; p>=1; --p) {
        const BounceData bounce1 = resGI.bounces[p - 1];

        const TriangleHit triangleHit1 = TriangleHit(bounce1.hit);
        const VertexData v1 = gScene.getVertexData(triangleHit1);

        const float3 wo1 = bounce1.wi;

        forwardUnwindStep(triangleHit1, v1, wo1, bounce1.sampledBsdfLobe, beam);
    }
    // Then we replay last bounce
    const float3 wo = bounce.wi;
    forwardUnwindStep(triangleHit, v, wo, bounce.sampledBsdfLobe, beam);
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= kOutputSize)) return;

    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);

    float3 finalColor = float3(0.f);

    const uint reservoirIdx = pixel.y * kOutputSize.x + pixel.x;
    const BounceData bounce = firstBounceBuffer[pixel.x + (pixel.y * kOutputSize.x)];

    const TriangleHit triangleHit = TriangleHit(bounce.hit);
    const VertexData v = gScene.getVertexData(triangleHit);

    let lod = ExplicitLodTextureSampler(0.f);
    const uint materialID = gScene.getMaterialID(triangleHit.instanceID);
    const ShadingData sd = gScene.materials.prepareShadingData(v, materialID, bounce.wi, lod);

    let bsdf = gScene.materials.getMaterialInstance(sd, lod);

    // DI
    {
        Reservoir reservoir = Reservoir::unpack(reservoirBuffer[reservoirIdx]);
        if (reservoir.M > 0) // check valid reservoir
        {
            const uint lobes = bsdf.getLobes(sd);

            const bool sampleReflection = (lobes & (uint)LobeType::NonDeltaReflection)!=0;
            const bool sampleTransmission = (lobes & (uint)LobeType::NonDeltaTransmission)!=0;

            LightSample ls = {};
            if (reservoir.hwss_comps > 1) {
                generateLightSampleFromStored(reservoir.sls, sampleReflection, sampleTransmission, sd, sg, gWavelengthsDI[reservoirIdx], ls);
            }
            else {
                generateLightSampleFromStored(reservoir.sls, sampleReflection, sampleTransmission, sd, sg, hwss_t(reservoir.heroWavelength, gWavelengthsDI[reservoirIdx].yzw), ls);
            }
            forwardUnwindDI(bounce, triangleHit, v, pixel, ls.beam);
            finalColor += measure(bounce.pathContribution, ls.beam) * reservoir.W;
        }
    }

    // GI
    ReservoirGI reservoirGI = ReservoirGI::unpack(reservoirGIBuffer[reservoirIdx]);
    if (reservoirGI.M > 0) // check valid reservoir
    {
        const BounceData bounceGI = reservoirGI.bounces[reservoirGI.bounceCount - 1];

        const TriangleHit triangleHitGI = TriangleHit(bounceGI.hit);
        const VertexData vGI = gScene.getVertexData(triangleHitGI);

        const uint materialIDGI = gScene.getMaterialID(triangleHitGI.instanceID);
        const ShadingData sdGI = gScene.materials.prepareShadingData(vGI, materialIDGI, bounceGI.wi, lod);

        let bsdfGI = gScene.materials.getMaterialInstance(sdGI, lod);

        const uint lobes = bsdfGI.getLobes(sdGI);

        const bool sampleReflection = (lobes & (uint)LobeType::NonDeltaReflection)!=0;
        const bool sampleTransmission = (lobes & (uint)LobeType::NonDeltaTransmission)!=0;

        LightSample ls = {};
        if (reservoirGI.hwss_comps > 1) {
            generateLightSampleFromStored(reservoirGI.sls, sampleReflection, sampleTransmission, sdGI, sg, gWavelengthsGI[reservoirIdx], ls);
        }
        else {
            generateLightSampleFromStored(reservoirGI.sls, sampleReflection, sampleTransmission, sdGI, sg, hwss_t(reservoirGI.heroWavelength, gWavelengthsGI[reservoirIdx].yzw), ls);
        }
        forwardUnwindGI(bounce, triangleHit, v, pixel, reservoirGI, ls.beam);
        finalColor += measure(bounce.pathContribution, ls.beam) * reservoirGI.W;
    }
    gOutputColor[pixel] = float4(finalColor, 1.f);
}
