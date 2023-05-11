#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "common.h"

layout(location = 0) out vec4 out_fragColor;

layout (location = 0 ) in VS_OUT
{
  vec2 texCoord;
} vOut;

layout(binding = 0, set = 0) uniform AppData
{
  UniformParams Params;
};

layout (binding = 1) uniform sampler2D shadowMap;
layout (binding = 2) uniform sampler2D gAlbedo;
layout (binding = 3) uniform sampler2D gNormals;
layout (binding = 4) uniform sampler2D gPos;
layout (binding = 5) uniform sampler2D indirectLight;

float dist(vec3 pos, vec3 N)
{
  const vec4 shrinkedPos = vec4(pos - 0.005 * N, 1.0);
  const vec4 shrinkedLightClipSpace = Params.lightMatrix * shrinkedPos;
  const vec3 shrinkedLightNDC = shrinkedLightClipSpace.xyz / shrinkedLightClipSpace.w;
  const vec2 shadowTexCoord = shrinkedLightNDC.xy * 0.5f + vec2(0.5f, 0.5f);
  const float depth = texture(shadowMap, shadowTexCoord).x;
  const vec4 sampledPosViewSpace = Params.lightProjInverse * vec4(shrinkedLightNDC.xy, depth, 1.0f);
  const float d1 = sampledPosViewSpace.z / sampledPosViewSpace.w;
  const vec4 shrinkedLightViewSpace = Params.lightProjInverse * shrinkedLightClipSpace;
  const float d2 = shrinkedLightViewSpace.z / shrinkedLightViewSpace.w;
  return abs(d1 - d2);
}

vec3 T(float s) {
  return vec3(0.233, 0.455, 0.649) * exp(-s*s/0.0064) +
         vec3(0.1, 0.336, 0.344) * exp(-s*s/0.0484) +
         vec3(0.118, 0.198, 0.0) * exp(-s*s/0.187) +
         vec3(0.113, 0.007, 0.007) * exp(-s*s/0.567) +
         vec3(0.358, 0.004, 0.0) * exp(-s*s/1.99) +
         vec3(0.078, 0.0, 0.0) * exp(-s*s/7.41);
}

void main()
{
  const vec4 wPos = texture(gPos, vOut.texCoord);
  const vec4 wNorm = texture(gNormals, vOut.texCoord);
  const vec4 posLightClipSpace = Params.lightMatrix * wPos; // 
  const vec3 posLightSpaceNDC  = posLightClipSpace.xyz / posLightClipSpace.w;    // for orto matrix, we don't need perspective division, you can remove it if you want; this is general case;
  const vec2 shadowTexCoord    = posLightSpaceNDC.xy * 0.5f + vec2(0.5f, 0.5f);  // just shift coords from [-1,1] to [0,1]               

  const float sampledDepth = texture(shadowMap, shadowTexCoord).x;
  const bool  outOfView = (shadowTexCoord.x < 0.0001f || shadowTexCoord.x > 0.9999f || shadowTexCoord.y < 0.0091f || shadowTexCoord.y > 0.9999f);
  const float shadow    = ((posLightSpaceNDC.z < sampledDepth + 0.001f) || outOfView) ? 1.0f : 0.0f;

  const vec4 lightColor = vec4(1.0f, 1.0f, 1.0f, 1.0f);
  const vec3 subsurfaceColor = vec3(0.4f, 0.2f, 0.1f);
  const vec3 lightDir   = normalize(Params.lightPos - wPos.xyz);
  const vec4 diffuse    = max(dot(wNorm.xyz, lightDir), 0.0f) * lightColor * Params.lightIntensity;

  const vec4 indirect   = Params.useIndirectLighting ? textureLod(indirectLight, vOut.texCoord, 0) : vec4(0.0);

  const vec4 albedo = texture(gAlbedo, vOut.texCoord);
  out_fragColor = (indirect + diffuse * shadow) * albedo;
  if (Params.useSss)
  {
    const float s = Params.sssScale * dist(wPos.xyz, wNorm.xyz);
    const float E = max(0.3 + dot(-wNorm.xyz, lightDir), 0.0);
    const vec3 transmittance = Params.sssAttenuation * T(s) * lightColor.xyz * Params.lightIntensity * albedo.xyz * E;
    out_fragColor += vec4(transmittance, 1.0);
  }
}