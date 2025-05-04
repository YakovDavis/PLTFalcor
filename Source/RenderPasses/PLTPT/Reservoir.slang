
#include "Utils/Math/MathConstants.slangh"

import Utils.Sampling.SampleGenerator;

// Since BounceData is in a separate buffer, we only need to store some additional info
struct PathInfo {
    // 4x4B
    float wavelength;               ///< Hero wavelength
    float weight;                   ///< Path backward weight estimate (via fully-coherent tracing)
    float pdf;                      ///< Path backward sampling pdf
    uint valid;                     ///< is valid path stored

    // 4x4B
    uint id;                        ///< Path ID encodes pixel with 16 bits each for pixel x|y
    uint frameCount;                ///< For SampleGenerator restoration for solve stage
    uint isPrevBounceBuffer;        ///< If >0 the bounce data is in prevBuffer
    uint _pad = 0;
}

struct PackedReservoir
{
    PathInfo pathInfo;
    uint W;
    uint M;
};

struct Reservoir {
    PathInfo pathInfo; ///< The stored path info.
    float weightSum; ///< The sum of weights.
    float W; ///< The weight of the reservoir.
    uint M; ///< The number of samples seen so far.

    __init()
    {
        this.pathInfo = {};
        this.weightSum = 0;
        this.W = 0;
        this.M = 0;
    }

    /*[mutating]
    bool update<S : ISampleGenerator>(const PathInfo newPathInfo, const float targetPDF, const float sourcePDF, inout S sg)
    {
        float weight = targetPDF / sourcePDF;
        this.weightSum += weight;
        this.M += 1;
        bool isSelected = sampleNext1D(sg) * this.weightSum < weight;

    	if(isSelected) {
    		this.pathInfo = newPathInfo;
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

        bool isSelected;
        if (!reservoir.pathInfo.valid)
        {
            isSelected = false;
        }
        else if (!this.pathInfo.valid)
        {
            isSelected = true;
        }
        else
        {
            isSelected = sampleNext1D(sg) * this.weightSum < weight;
        }

        if (isSelected) {
            this.pathInfo = reservoir.pathInfo;
            this.W = targetPDF;
        }

        return isSelected;
    }

    static Reservoir unpack(const PackedReservoir packedReservoir)
    {
        Reservoir reservoir;

        reservoir.pathInfo = packedReservoir.pathInfo;
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
        packedReservoir.pathInfo = this.pathInfo;
        packedReservoir.W = asuint(this.W);
        packedReservoir.M = this.M;
        return packedReservoir;
    }
}
