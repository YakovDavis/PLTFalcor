
#include "Utils/Math/MathConstants.slangh"

import Path;
import Utils.Sampling.SampleGenerator;
import Rendering.PLT;

struct PackedReservoir
{
    // 40B
    BounceData bounceData;

    // 4x4B
    uint W;
    uint M;
    uint beamCount;
    uint _pad;
};

struct Reservoir {
    BounceData bounceData;
    float weightSum; ///< The sum of weights.
    float W; ///< The weight of the reservoir.
    uint M; ///< The number of samples seen so far.
    uint beamCount;

    __init()
    {
        this.bounceData = {};
        this.weightSum = 0;
        this.W = 0;
        this.M = 0;
        this.beamCount = 0;
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

        reservoir.bounceData = packedReservoir.bounceData;

        reservoir.W = asfloat(packedReservoir.W);
        reservoir.M = packedReservoir.M;
        reservoir.weightSum = reservoir.M * reservoir.W;

        if (isinf(reservoir.W) || isnan(reservoir.W))
        {
            reservoir.W = 0.f;
            reservoir.M = 0;
        }

        reservoir.beamCount = packedReservoir.beamCount;

        return reservoir;
    }

    PackedReservoir pack()
    {
        PackedReservoir packedReservoir = {};
        packedReservoir.bounceData = this.bounceData;
        packedReservoir.W = asuint(this.W);
        packedReservoir.M = this.M;
        packedReservoir.beamCount = this.beamCount;
        return packedReservoir;
    }
}
