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

void main()
{
  const vec4 wPos = texture(gPos, vOut.texCoord);
  const vec4 wNorm = texture(gNormals, vOut.texCoord);
  const vec4 posLightClipSpace = Params.lightMatrix * wPos; // 
  const vec3 posLightSpaceNDC  = posLightClipSpace.xyz / posLightClipSpace.w;    // for orto matrix, we don't need perspective division, you can remove it if you want; this is general case;
  const vec2 shadowTexCoord    = posLightSpaceNDC.xy * 0.5f + vec2(0.5f, 0.5f);  // just shift coords from [-1,1] to [0,1]               
    
  const bool  outOfView = (shadowTexCoord.x < 0.0001f || shadowTexCoord.x > 0.9999f || shadowTexCoord.y < 0.0091f || shadowTexCoord.y > 0.9999f);
  const float shadow    = ((posLightSpaceNDC.z < texture(shadowMap, shadowTexCoord).x + 0.001f) || outOfView) ? 1.0f : 0.0f;

  const vec4 lightColor = vec4(1.0f, 1.0f, 1.0f, 1.0f);
  const vec3 lightDir   = normalize(Params.lightPos - wPos.xyz);
  const vec4 diffuse    = max(dot(wNorm.xyz, lightDir), 0.0f) * lightColor * Params.lightIntensity;

  const vec4 indirect   = Params.useIndirectLighting ? textureLod(indirectLight, vOut.texCoord, 0) : vec4(0.0);

  out_fragColor = (indirect + diffuse * shadow) * texture(gAlbedo, vOut.texCoord);
}