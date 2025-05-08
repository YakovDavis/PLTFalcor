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

// Unwind last step before measure
void unwindLastStep(const BounceData bounce, const uint2 pixel, inout PLTBeam beam) {

    const TriangleHit triangleHit = TriangleHit(bounce.hit);
    const VertexData v = gScene.getVertexData(triangleHit);

    const float3 wo = bounce.wi;

    let lod = ExplicitLodTextureSampler(0.f);
    const uint materialID = gScene.getMaterialID(triangleHit.instanceID);
    const ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -beam.r, lod);

    // Propagate
    beam.propagate(sd.posW);

    // BSDF
    let bsdf = gScene.materials.getMaterialInstance(sd, lod);
    bsdf.eval(sd, wo, bounce.sampledBsdfLobe, beam);

    // Spectral MIS
    forwardHWSSMis(bsdf, sd, wo, bounce.sampledBsdfLobe, beam);
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

uniform uint gMaxBeamCount;
uniform uint2 gFrameDim;

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    float3 finalColor = float3(0.f);

    const uint2 pixel = dispatchThreadId.xy;
    const uint beam0Idx = (pixel.x + (pixel.y * gFrameDim.x)) * gMaxBeamCount;

    const uint reservoirIdx = pixel.y * gFrameDim.x + pixel.x;
    const Reservoir reservoir = Reservoir::unpack(reservoirBuffer[reservoirIdx]);
    const BounceData bounce = reservoir.bounceData;

    for (int p = 0; p < gMaxBeamCount; ++p)
    {
        PLTBeam beam = PLTBeam::unpack(beamBuffer[beam0Idx + p]);
        unwindLastStep(bounce, pixel, beam);
        finalColor += measure(bounce.pathContribution, beam);
    }
    gOutputColor[pixel] = float4(finalColor, 1.f); // TODO: handle emissives
}
