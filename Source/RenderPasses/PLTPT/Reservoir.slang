
#include "Utils/Math/MathConstants.slangh"

import Utils.HWSS;
import Path;
import Utils.Sampling.SampleGenerator;
import Rendering.PLT;

// 112 bytes
struct PackedReservoir
{
    // 2x16B
    float4 posW_W;
    float4 r_M;

    // 4x8B
    half4 L;
    half4 L1;
    half4 L2;
    half4 L3;

    // 2x8B
    half4 wavelengths;
    half4 spectralPDFs;

    // 16B
    float4 packedTheta;

    // 16B
    float4 t_distant_hwss_comps;
};

struct Reservoir {
    float weightSum; ///< The sum of weights.
    float W; ///< The weight of the reservoir.
    uint M; ///< The number of samples seen so far.

    hwss_t L = .0f;
    hwss_t L1 = .0f,L2 = .0f,L3 = .0f;

    // Wavelengths
    hwss_t wavelengths;             ///< Selected wavelengths
    hwss_t spectralPDFs;
    uint hwss_comps = hwss_samples;

    // Coherence
    PLTCoherence Theta;

    float3 posW;
    float3 r;                   ///< Direction of propagation
    float3 t;                   ///< Beam tangent (direction of horizontal linear polarization)
    bool distant = false;

    __init()
    {
        this.weightSum = 0;
        this.W = 0;
        this.M = 0;

        this.hwss_comps = hwss_samples;
        this.wavelengths = hwss_t(.0f);
        this.spectralPDFs = 1.f;

        this.L = hwss_t(.0f);
        this.L1 = this.L2 = this.L3 = hwss_t(.0f);

        this.Theta = PLTCoherence(0.025, 1e-3f);

        this.posW = float3(.0f);
        this.r = float3(.0f);
        this.t = float3(.0f, .78f, -.63f);
        this.distant = true;
    }

    /*[mutating]
    bool update<S : ISampleGenerator>(const PLTBeam newPltBeam, const float targetPDF, const float sourcePDF, inout S sg)
    {
        float weight = targetPDF / sourcePDF;
        this.weightSum += weight;
        this.M += 1;
        bool isSelected = sampleNext1D(sg) * this.weightSum < weight;

        if (isSelected) {
            this.W = targetPDF;
            this.posW = newPltBeam.posW;
            this.r = newPltBeam.r;
            this.distant = newPltBeam.distant;
            this.Theta = newPltBeam.Theta;
            this.t = newPltBeam.t;
            this.L = newPltBeam.L;
            this.L1 = newPltBeam.L1;
            this.L2 = newPltBeam.L2;
            this.L3 = newPltBeam.L3;
            this.wavelengths = newPltBeam.wavelengths;
            this.spectralPDFs = newPltBeam.spectralPDFs;
            this.hwss_comps = newPltBeam.hwss_comps;
        }

        return isSelected;
    }*/

    [mutating]
    bool update<S : ISampleGenerator>(inout Reservoir reservoir, const float targetPDF, inout S sg)
    {
        float weight = targetPDF * reservoir.W * reservoir.M;
        this.weightSum += weight;
        this.M += reservoir.M;

        bool isSelected = sampleNext1D(sg) * this.weightSum < weight;

        if (isSelected) {
            this.W = targetPDF;
            this.posW = reservoir.posW;
            this.r = reservoir.r;
            this.distant = reservoir.distant;
            this.Theta = reservoir.Theta;
            this.t = reservoir.t;
            this.L = reservoir.L;
            this.L1 = reservoir.L1;
            this.L2 = reservoir.L2;
            this.L3 = reservoir.L3;
            this.wavelengths = reservoir.wavelengths;
            this.spectralPDFs = reservoir.spectralPDFs;
            this.hwss_comps = reservoir.hwss_comps;
        }

        return isSelected;
    }

    PLTBeam getBeam()
    {
        PLTBeam beam;

        beam.L = this.L;
        beam.L1 = this.L1;
        beam.L2 = this.L2;
        beam.L3 = this.L3;

        beam.wavelengths = this.wavelengths;
        beam.spectralPDFs = this.spectralPDFs;
        beam.hwss_comps = this.hwss_comps;

        beam.posW = this.posW;
        beam.r = this.r;
        beam.distant = this.distant;
        beam.Theta = this.Theta;
        beam.t = this.t;

        return beam;
    }

    float getRestirPdf()
    {
        float result = 0.f;
        for (int p = 0; p < hwss_comps; ++p)
        {
            result += L[p];
        }
        return result;
    }

    static Reservoir unpack(const PackedReservoir packedReservoir)
    {
        Reservoir reservoir;

        reservoir.W = packedReservoir.posW_W.w;
        reservoir.M = asuint(packedReservoir.r_M.w);
        reservoir.weightSum = reservoir.M * reservoir.W;

        if (isinf(reservoir.W) || isnan(reservoir.W))
        {
            reservoir.W = 0.f;
            reservoir.M = 0;
        }

        reservoir.posW = packedReservoir.posW_W.xyz;
        reservoir.r = packedReservoir.r_M.xyz;

        reservoir.L = hwss_t(0.f);
        reservoir.L[0] = float(packedReservoir.L.x); reservoir.L[1] = float(packedReservoir.L.y); reservoir.L[2] = float(packedReservoir.L.z); reservoir.L[3] = float(packedReservoir.L.w);
        reservoir.L1 = hwss_t(0.f);
        reservoir.L1[0] = float(packedReservoir.L1.x); reservoir.L1[1] = float(packedReservoir.L1.y); reservoir.L1[2] = float(packedReservoir.L1.z); reservoir.L1[3] = float(packedReservoir.L1.w);
        reservoir.L2 = hwss_t(0.f);
        reservoir.L2[0] = float(packedReservoir.L2.x); reservoir.L2[1] = float(packedReservoir.L2.y); reservoir.L2[2] = float(packedReservoir.L2.z); reservoir.L2[3] = float(packedReservoir.L2.w);
        reservoir.L3 = hwss_t(0.f);
        reservoir.L3[0] = float(packedReservoir.L3.x); reservoir.L3[1] = float(packedReservoir.L3.y); reservoir.L3[2] = float(packedReservoir.L3.z); reservoir.L3[3] = float(packedReservoir.L3.w);

        reservoir.wavelengths = hwss_t(0.f);
        reservoir.wavelengths[0] = float(packedReservoir.wavelengths.x); reservoir.wavelengths[1] = float(packedReservoir.wavelengths.y); reservoir.wavelengths[2] = float(packedReservoir.wavelengths.z); reservoir.wavelengths[3] = float(packedReservoir.wavelengths.w);
        reservoir.spectralPDFs = hwss_t(0.f);
        reservoir.spectralPDFs[0] = float(packedReservoir.spectralPDFs.x); reservoir.spectralPDFs[1] = float(packedReservoir.spectralPDFs.y); reservoir.spectralPDFs[2] = float(packedReservoir.spectralPDFs.z); reservoir.spectralPDFs[3] = float(packedReservoir.spectralPDFs.w);

        reservoir.Theta.D = float3(packedReservoir.packedTheta.xyz);
        reservoir.Theta.r_m = float(packedReservoir.packedTheta.w);

        reservoir.t = float3(packedReservoir.t_distant_hwss_comps.xyz);
        uint distant_hwss_comps = asuint(packedReservoir.t_distant_hwss_comps.w);
        reservoir.hwss_comps = distant_hwss_comps > 0u ? 4 : 0; // distant_hwss_comps & 0x0000FFFFu; TODO: Fix, idk why this crashes
        reservoir.distant = (distant_hwss_comps >> 16) > 0u;

        return reservoir;
    }

    PackedReservoir pack()
    {
        PackedReservoir packedReservoir = {};
        packedReservoir.posW_W = float4(this.posW, this.W);
        packedReservoir.r_M = float4(this.r, asfloat(this.M));
        packedReservoir.L = half4(half(this.L[0]), half(this.L[1]), half(this.L[2]), half(this.L[3]));
        packedReservoir.L1 = half4(half(this.L1[0]), half(this.L1[1]), half(this.L1[2]), half(this.L1[3]));
        packedReservoir.L2 = half4(half(this.L2[0]), half(this.L2[1]), half(this.L2[2]), half(this.L2[3]));
        packedReservoir.L3 = half4(half(this.L3[0]), half(this.L3[1]), half(this.L3[2]), half(this.L3[3]));

        packedReservoir.wavelengths = half4(half(this.wavelengths[0]), half(this.wavelengths[1]), half(this.wavelengths[2]), half(this.wavelengths[3]));
        packedReservoir.spectralPDFs = half4(half(this.spectralPDFs[0]), half(this.spectralPDFs[1]), half(this.spectralPDFs[2]), half(this.spectralPDFs[3]));

        packedReservoir.packedTheta = float4(this.Theta.D.x, this.Theta.D.y, this.Theta.D.z, this.Theta.r_m);

        uint distant_hwss_comps = (distant ? 0x00010000u : 0u) | (hwss_comps & 0x0000FFFFu);
        packedReservoir.t_distant_hwss_comps = float4(this.t.x, this.t.y, this.t.z, asfloat(distant_hwss_comps));

        return packedReservoir;
    }
}
