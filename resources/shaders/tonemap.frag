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
} params;

layout(binding = 0) uniform sampler2D hdrImage;

vec3 tonemap(vec3 hdrColor)
{
  return hdrColor;
}

void main() {
  vec3 color = texture(hdrImage, vOut.texCoord).xyz;
  out_fragColor = params.tonemapEnabled ? vec4(tonemap(color), 1.0f)
                                        : vec4(clamp(color, 0.0f, 1.0f), 1.0f);
}