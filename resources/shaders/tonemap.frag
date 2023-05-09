#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "common.h"

layout(location = 0) out vec4 out_fragColor;

layout (location = 0) in VS_OUT
{
  vec2 texCoord;
} vOut;

layout(push_constant) uniform tonemapParams
{
  bool tonemapEnabled;
  uint tonemappingCurve;
} params;

layout(binding = 0) uniform sampler2D hdrImage;

const mat3 ACESInputMat =
{
  {0.59719, 0.35458, 0.04823},
  {0.07600, 0.90834, 0.01566},
  {0.02840, 0.13383, 0.83777}
};

const mat3 ACESOutputMat =
{
  { 1.60475, -0.53108, -0.07367},
  {-0.10208,  1.10813, -0.00605},
  {-0.00327, -0.07276,  1.07602}
};

vec3 RRTAndODTFit(vec3 v)
{
  vec3 a = v * (v + 0.0245786f) - 0.000090537f;
  vec3 b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
  return a / b;
}

vec3 hillAces(vec3 col)
{
  vec3 c = ACESInputMat * col;
  c = RRTAndODTFit(c);
  return clamp(c, 0.0f, 1.0f);
}

vec3 narkowiczAces(vec3 col)
{
  vec3 res = (col * (2.51f * col + 0.03f)) / (col * (2.43f * col + 0.59f) + 0.14f);
  return clamp(res, 0.0f, 1.0f);
}

void main() {
  vec3 color = texture(hdrImage, vOut.texCoord).xyz;
  if (params.tonemapEnabled)
  {
    switch (params.tonemappingCurve)
    {
      case 0:
        out_fragColor = vec4(hillAces(color), 1.0f);
        break;
      case 1:
        out_fragColor = vec4(narkowiczAces(color), 1.0f);
        break;
    }
  }
  else
  {
    out_fragColor = vec4(clamp(color, 0.0f, 1.0f), 1.0f);
  }
}