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
  color = texture(image, surf.texCoord);
  color.a *= drawData[int(surf.idx)].opacity;
}
