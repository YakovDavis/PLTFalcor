
/** Describes a minimal light sample to store in reservoir and resample */
struct StoredLightSample
{
    float pdf;
    uint lightType;
    uint triangleOrLightIndex;
    float2 uvOrEnvSample; // for emissive
}
