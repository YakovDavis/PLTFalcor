

/** Describes a minimal light sample to store in reservoir and resample */
struct StoredLightSample
{
    // 7x4B
    float pdf;
    uint16_t lightType;
    uint16_t triangleOrLightIndex;
    float2 uvOrEnvSample; // for emissive
    float3 posW; // for bsdf eval and visibility
}
