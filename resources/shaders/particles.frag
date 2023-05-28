#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "common.h"

layout(location = 0) out vec4 color;

layout(std430, binding = 0) buffer particleDrawListBuffer
{
    ParticleDrawData drawData[];
};

layout(binding = 1) uniform AppData
{
    UniformParams Params;
};

layout (binding = 2) uniform sampler2D depth;
layout (binding = 3) uniform sampler2D image;
layout (binding = 4) uniform sampler2D perlin2D;

layout (location = 0) in VS_OUT
{
  vec2 texCoord;
  float idx;
} surf;

void main()
{
  vec2 screenTexCoord = vec2(gl_FragCoord.x / Params.screenWidth, gl_FragCoord.y / Params.screenHeight);
  if (texture(depth, screenTexCoord).x < gl_FragCoord.z)
    discard;

  const float opacity = drawData[int(surf.idx)].opacity;
  color = texture(image, surf.texCoord + mix(0.7, 0, opacity) * vec2(texture(perlin2D, surf.texCoord)));
  color.a *= opacity;
}
