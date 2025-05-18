
#include "Utils/Math/MathConstants.slangh"

import Scene.ShadingData;
import Utils.Sampling.TinyUniformSampleGenerator;
import Rendering.Materials.BxDF;
import Utils.Color.ColorHelpers;
import Rendering.Materials.Microfacet;
import Rendering.Materials.Fresnel;
import Utils.Math.PackedFormats;
import Utils.Geometry.GeometryHelpers;

__exported import Rendering.Materials.IMaterialInstance;

/** A structure representing the packed information related to a surface point
 */
struct PackedSurfaceData ///< 32 bytes
{
    uint3 position;         ///< Packed position of the surface point in the world space.
    uint normal;            ///< Packed normal vector at the surface point.
    uint depth;             ///< Packed depth (distance from a camera) at the surface point.
    uint3 _pad = uint3(0);
};

/** A structure that represents the position, normal, and reflectance properties of a surface point in a scene.
 */
struct SurfaceData
{
    float3 position; ///< The position of the surface point in the world space.
    float3 normal;   ///< The normal vector at the surface point.
    float depth;             ///< The depth (distance from a camera) at the surface point.

    /** Creates an invalid SurfaceData object, used as a sentinel value.
     */
    static SurfaceData createInvalid()
    {
        SurfaceData surfaceData = {};
        surfaceData.depth = -FLT_MAX;
        return surfaceData;
    }

    /** Creates a SurfaceData object from given shading data, bsdf, and camera position.
     */
    static SurfaceData create(const ShadingData sd, const IMaterialInstance bsdf, const float3 cameraPosition)
    {
        SurfaceData surfaceData;
        surfaceData.position = sd.posW;
        surfaceData.normal = sd.N;
        surfaceData.depth = distance(sd.posW, cameraPosition);

        return surfaceData;
    }

    /** Unpacks a PackedSurfaceData object to create a SurfaceData object.
     */
    static SurfaceData unpack(const PackedSurfaceData packedSurfaceData)
    {
        SurfaceData surfaceData;
        surfaceData.position = asfloat(packedSurfaceData.position);
        surfaceData.normal = decodeNormal2x16(packedSurfaceData.normal);
        surfaceData.depth = asfloat(packedSurfaceData.depth);

        return surfaceData;
    }

    /** Packs the SurfaceData object into a PackedSurfaceData object for more efficient storage and transfer.
     */
    PackedSurfaceData pack()
    {
        PackedSurfaceData packedSurfaceData = {};
        packedSurfaceData.position = asuint(this.position);
        packedSurfaceData.normal = encodeNormal2x16(this.normal);
        packedSurfaceData.depth = asuint(this.depth);

        return packedSurfaceData;
    }

    /** Returns true if the SurfaceData is valid, i.e., if it corresponds to a real surface point in the scene.
     *  The SurfaceData is considered invalid if its depth is -FLT_MAX, a sentinel value.
     */
    bool isValid()
    {
        return depth != -FLT_MAX;
    }
};
