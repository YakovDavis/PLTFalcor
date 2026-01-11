
#include "Utils/Math/MathConstants.slangh"

import Rendering.Lights.StoredLightSample;

import Utils.HWSS;
import Path;
import Utils.Sampling.SampleGenerator;
import Rendering.PLT;

// 128 bytes
struct PackedReservoirGI
{
    // 4x4B
    float W;
    uint M;
    uint bounceCount;
    float cachedIntensity;

    // 8x4B
    StoredLightSample sls;
    uint16_t hwss_comps;
    float16_t heroWavelength;

    // 2x40B
    BounceData bounces[2];
};

struct ReservoirGI {
    float weightSum; ///< The sum of weights.
    float W;         ///< The weight of the reservoir.
    uint M;          ///< The number of samples seen so far.

    uint bounceCount;

    uint hwss_comps;
    float cachedIntensity; ///< Average intensity between wavelength of the initial beam
    float heroWavelength;

    StoredLightSample sls;

    BounceData bounces[2];

    __init()
    {
        this.weightSum = 0;
        this.W = 0;
        this.M = 0;

        this.bounceCount = 0;
        this.hwss_comps = 0;

        this.cachedIntensity = 0;

        this.heroWavelength = 0;

        this.bounces[0] = {};
        this.bounces[1] = {};

        this.sls = {};
    }

    /*[mutating]
    bool update<S : ISampleGenerator>(const PackedPLTBeam newPltBeam, const float targetPDF, const float sourcePDF, inout S sg)
    {
        float weight = targetPDF / sourcePDF;
        this.weightSum += weight;
        this.M += 1;
        bool isSelected = sampleNext1D(sg) * this.weightSum < weight;

    	if(isSelected) {
    		this.W = targetPDF;
    	}

    	return isSelected;
    }*/

    [mutating]
    bool update<S : ISampleGenerator>(inout ReservoirGI reservoir, const float targetPDF, inout S sg)
    {
        if (targetPDF == 0 || reservoir.W == 0 || reservoir.M == 0) {
            return false;
        }
        float weight = targetPDF * reservoir.W * reservoir.M;
        this.weightSum += weight;
        this.M += reservoir.M;

        bool isSelected = sampleNext1D(sg) * this.weightSum < weight;

        if (isSelected) {
            this.W = targetPDF;
            this.bounceCount = reservoir.bounceCount;
            this.hwss_comps = reservoir.hwss_comps;

            this.cachedIntensity = reservoir.cachedIntensity;
            this.heroWavelength = reservoir.heroWavelength;

            this.bounces[0] = reservoir.bounces[0];
            this.bounces[1] = reservoir.bounces[1];

            this.sls = reservoir.sls;
        }

        return isSelected;
    }

    static ReservoirGI unpack(const PackedReservoirGI packedReservoir)
    {
        ReservoirGI reservoir;

        reservoir.W = packedReservoir.W;
        reservoir.M = packedReservoir.M;
        reservoir.weightSum = reservoir.M * reservoir.W;

        if (isinf(reservoir.W) || isnan(reservoir.W))
        {
            reservoir.W = 0.f;
            reservoir.M = 0;
        }

        reservoir.bounceCount = packedReservoir.bounceCount;
        if (reservoir.bounceCount > 2)
        {
            reservoir.bounceCount = 2;
        }
        reservoir.hwss_comps = packedReservoir.hwss_comps;
        reservoir.cachedIntensity = packedReservoir.cachedIntensity;
        reservoir.heroWavelength = packedReservoir.heroWavelength;

        reservoir.sls = packedReservoir.sls;

        reservoir.bounces[0] = packedReservoir.bounces[0];
        reservoir.bounces[1] = packedReservoir.bounces[1];

        return reservoir;
    }

    PackedReservoirGI pack()
    {
        PackedReservoirGI packedReservoir = {};
        packedReservoir.W = this.W;
        packedReservoir.M = this.M;

        packedReservoir.bounceCount = this.bounceCount;
        packedReservoir.hwss_comps = this.hwss_comps;
        packedReservoir.cachedIntensity = this.cachedIntensity;
        packedReservoir.heroWavelength = this.heroWavelength;

        packedReservoir.sls = this.sls;

        packedReservoir.bounces[0] = this.bounces[0];
        packedReservoir.bounces[1] = this.bounces[1];
        return packedReservoir;
    }
}
