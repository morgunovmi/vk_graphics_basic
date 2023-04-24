#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "common.h"

layout(location = 0) out vec4 out_fragColor;

layout (location = 0 ) in VS_OUT
{
  vec2 texCoord;
} vOut;

layout(push_constant) uniform params_t
{
    mat4 projInverse;
    mat4 viewInverse;
} params;

layout(binding = 0, set = 0) uniform AppData
{
  UniformParams Params;
};

layout (binding = 1) uniform sampler2D shadowMap;
layout (binding = 2) uniform sampler2D gAlbedo;
layout (binding = 3) uniform sampler2D gNormals;
layout (binding = 4) uniform sampler2D gPosition;
layout (binding = 5) uniform sampler2D ssao;

void main()
{
  const vec4 vPos = textureLod(gPosition, vOut.texCoord, 0);
  const vec4 wPos = params.viewInverse * vPos;
  const vec4 vNorm = textureLod(gNormals, vOut.texCoord, 0);
  const vec4 wNorm = params.viewInverse * vNorm;

  const vec4 posLightClipSpace = Params.lightMatrix * wPos;
  const vec3 posLightSpaceNDC  = posLightClipSpace.xyz / posLightClipSpace.w;    // for orto matrix, we don't need perspective division, you can remove it if you want; this is general case;
  const vec2 shadowTexCoord    = posLightSpaceNDC.xy * 0.5f + vec2(0.5f, 0.5f);  // just shift coords from [-1,1] to [0,1]               
    
  const bool  outOfView = (shadowTexCoord.x < 0.0001f || shadowTexCoord.x > 0.9999f || shadowTexCoord.y < 0.0091f || shadowTexCoord.y > 0.9999f);
  const float shadow    = ((posLightSpaceNDC.z < textureLod(shadowMap, shadowTexCoord, 0).x + 0.001f) || outOfView) ? 1.0f : 0.0f;

  const vec4 lightColor = vec4(1.0f, 1.0f, 1.0f, 1.0f);
  const vec3 lightDir   = normalize(Params.lightPos - wPos.xyz);
  const vec4 diffuse    = max(dot(wNorm.xyz, lightDir), 0.0f) * lightColor;

  const float occlusion = Params.ssaoEnabled ? textureLod(ssao, vOut.texCoord, 0).x : 1.0f;
  const vec4 ambient    = 0.2f * vec4(Params.baseColor, 1.0f) * occlusion;

  out_fragColor = (ambient + diffuse * shadow) * textureLod(gAlbedo, vOut.texCoord, 0);
}