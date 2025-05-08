
#include "Utils/Math/MathConstants.slangh"

import Utils.Sampling.SampleGenerator;
import Rendering.PLT;

struct PackedReservoir
{
    uint W;
    uint M;
};

struct Reservoir {
    float weightSum; ///< The sum of weights.
    float W; ///< The weight of the reservoir.
    uint M; ///< The number of samples seen so far.

    __init()
    {
        this.weightSum = 0;
        this.W = 0;
        this.M = 0;
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
    bool update<S : ISampleGenerator>(inout Reservoir reservoir, const float targetPDF, inout S sg)
    {
        float weight = targetPDF * reservoir.W * reservoir.M;
        this.weightSum += weight;
        this.M += reservoir.M;

        bool isSelected = sampleNext1D(sg) * this.weightSum < weight;

        if (isSelected) {
            this.W = targetPDF;
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

        return reservoir;
    }

    PackedReservoir pack()
    {
        PackedReservoir packedReservoir = {};
        packedReservoir.W = asuint(this.W);
        packedReservoir.M = this.M;
        return packedReservoir;
    }
}
