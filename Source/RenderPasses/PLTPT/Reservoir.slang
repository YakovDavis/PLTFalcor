
#include "Utils/Math/MathConstants.slangh"

import Rendering.Lights.StoredLightSample;

import Utils.HWSS;
import Path;
import Utils.Sampling.SampleGenerator;
import Rendering.PLT;

// 48 bytes
struct PackedReservoir
{
    // 2x4B
    uint W;
    uint M;

    // 8x4B
    StoredLightSample sls;
    uint pad_ = 0;

    // 2x4B
    uint hwss_comps;
    float cachedIntensity;
};

struct Reservoir {
    float weightSum; ///< The sum of weights.
    float W; ///< The weight of the reservoir.
    uint M; ///< The number of samples seen so far.
    uint hwss_comps;
    float cachedIntensity; ///< Average intensity between wavelength of the initial beam

    StoredLightSample sls;

    __init()
    {
        this.weightSum = 0;
        this.W = 0;
        this.M = 0;
        this.hwss_comps = 0;
        cachedIntensity = 0;

        this.sls = {};
    }

    /*[mutating]
    bool update<S : ISampleGenerator>(const StoredLightSample newStoredLightSample, const float targetPDF, const float sourcePDF, inout S sg)
    {
        float weight = targetPDF / sourcePDF;
        this.weightSum += weight;
        this.M += 1;
        bool isSelected = sampleNext1D(sg) * this.weightSum < weight;

        if (isSelected) {
            this.W = targetPDF;
            this.sls = newStoredLightSample;
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
            this.sls = reservoir.sls;
            this.hwss_comps = reservoir.hwss_comps;
            this.cachedIntensity = reservoir.cachedIntensity;
        }

        return isSelected;
    }

    static Reservoir unpack(const PackedReservoir packedReservoir)
    {
        Reservoir reservoir;

        reservoir.W = asfloat(packedReservoir.W);
        reservoir.M = packedReservoir.M;
        reservoir.weightSum = reservoir.M * reservoir.W;

        if (isinf(reservoir.W) || isnan(reservoir.W))
        {
            reservoir.W = 0.f;
            reservoir.M = 0;
        }

        reservoir.hwss_comps = packedReservoir.hwss_comps;
        reservoir.cachedIntensity = packedReservoir.cachedIntensity;

        reservoir.sls = packedReservoir.sls;

        return reservoir;
    }

    PackedReservoir pack()
    {
        PackedReservoir packedReservoir = {};
        packedReservoir.W = asuint(this.W);
        packedReservoir.M = this.M;
        packedReservoir.hwss_comps = this.hwss_comps;
        packedReservoir.cachedIntensity = this.cachedIntensity;

        packedReservoir.sls = this.sls;

        return packedReservoir;
    }
}
