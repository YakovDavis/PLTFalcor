
#include "Utils/Math/MathConstants.slangh"

import Path;
import Utils.Sampling.SampleGenerator;
import Rendering.PLT;

// 48 bytes
struct PackedReservoirGI
{
    // 4x4B
    float W;
    uint M;
    uint beamCount;
    float lastHitNormal_x;

    // 2x16B
    float4 posW_lastHitNormal_y;
    float4 r_lastHitNormal_z;
};

struct ReservoirGI {
    float weightSum; ///< The sum of weights.
    float W;         ///< The weight of the reservoir.
    uint M;          ///< The number of samples seen so far.
    uint beamCount;

    float3 posW;
    float3 r;
    float3 lastHitNormal;

    __init()
    {
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
    bool update<S : ISampleGenerator>(inout ReservoirGI reservoir, const float targetPDF, inout S sg)
    {
        float weight = targetPDF * reservoir.W * reservoir.M;
        this.weightSum += weight;
        this.M += reservoir.M;

        bool isSelected = sampleNext1D(sg) * this.weightSum < weight;

        if (isSelected) {
            this.W = targetPDF;
            this.beamCount = reservoir.beamCount;
            this.posW = reservoir.posW;
            this.r = reservoir.r;
            this.lastHitNormal = reservoir.lastHitNormal;
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

        reservoir.beamCount = packedReservoir.beamCount;

        reservoir.posW = packedReservoir.posW_lastHitNormal_y.xyz;
        reservoir.r = packedReservoir.r_lastHitNormal_z.xyz;
        reservoir.lastHitNormal = float3(packedReservoir.lastHitNormal_x, packedReservoir.posW_lastHitNormal_y.w, packedReservoir.r_lastHitNormal_z.w);

        return reservoir;
    }

    PackedReservoirGI pack()
    {
        PackedReservoirGI packedReservoir = {};
        packedReservoir.W = this.W;
        packedReservoir.M = this.M;
        packedReservoir.beamCount = this.beamCount;
        packedReservoir.lastHitNormal_x = this.lastHitNormal.x;
        packedReservoir.posW_lastHitNormal_y = float4(this.posW, this.lastHitNormal.y);
        packedReservoir.r_lastHitNormal_z = float4(this.r, this.lastHitNormal.z);
        return packedReservoir;
    }
}
