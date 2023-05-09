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

layout (binding = 1) uniform sampler2D gNormals;
layout (binding = 2) uniform sampler2D gPos;
layout (binding = 3) uniform sampler2D rsmPos;
layout (binding = 4) uniform sampler2D rsmNorm;
layout (binding = 5) uniform sampler2D rsmFlux;

layout(binding = 6, set = 0) buffer RsmSamples
{
  vec2 samples[];
};

const float rsmRMax = 0.18;
const uint numRsmSamples = 400;
const float indirectIntensity = 0.05;

const float PI = 3.1415926538;

vec3 calcRsm(vec2 shadowTexCoord, vec3 wNorm, vec3 wPos)
{
  vec3 indirectLight = vec3(0.0);

  for (uint i = 0; i < numRsmSamples; ++i)
  {
    const vec2 rnd = samples[i];
    const vec2 coords = shadowTexCoord + rsmRMax * rnd.x * vec2(sin(2 * PI * rnd.y), cos(2 * PI * rnd.y));

    const vec3 wPosRsm  = texture(rsmPos, coords).xyz;
    const vec3 wNormRsm = texture(rsmNorm, coords).xyz;
    const vec3 flux     = texture(rsmFlux, coords).xyz;

    indirectLight += flux * rnd.x * rnd.x
      * max(0, dot(wNormRsm, wPos - wNormRsm))
      * max(0, dot(wNorm, wPosRsm - wPos))
      / pow(length(wPos - wPosRsm), 4);
  }

  return clamp(indirectLight * indirectIntensity, 0.0, 1.0);
}

void main()
{
  vec4 wPos = texture(gPos, vOut.texCoord);
  const vec4 wNorm = texture(gNormals, vOut.texCoord);
  const vec4 posLightClipSpace = Params.lightMatrix * wPos; // 
  const vec3 posLightSpaceNDC  = posLightClipSpace.xyz / posLightClipSpace.w;    // for orto matrix, we don't need perspective division, you can remove it if you want; this is general case;
  const vec2 shadowTexCoord    = posLightSpaceNDC.xy * 0.5f + vec2(0.5f, 0.5f);  // just shift coords from [-1,1] to [0,1]               

  out_fragColor = vec4(calcRsm(shadowTexCoord, wNorm.xyz, wPos.xyz), 1.0);
}