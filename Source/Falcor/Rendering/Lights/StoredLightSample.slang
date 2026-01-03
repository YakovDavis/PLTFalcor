

/** Describes a minimal light sample to store in reservoir and resample */
struct StoredLightSample
{
    // 7x4B
    float16_t pdf;
    uint16_t distant;
    uint16_t lightType;
    uint16_t triangleOrLightIndex;
    float2 uvOrEnvSample; // for emissive
    float3 posW; // for bsdf eval and visibility, for distant just direction multiplied by large number
}
