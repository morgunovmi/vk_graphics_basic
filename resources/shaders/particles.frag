#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "common.h"

layout(location = 0) out vec4 color;

layout(binding = 1) uniform AppData
{
    UniformParams Params;
};

layout (binding = 2) uniform sampler2D depth;

layout (location = 0 ) in VS_OUT
{
  vec2 texCoord;
} surf;

void main()
{
  vec2 screenTexCoord = vec2(gl_FragCoord.x / Params.screenWidth, gl_FragCoord.y / Params.screenHeight);
  if (texture(depth, screenTexCoord).x < gl_FragCoord.z)
    discard;
  color = vec4(1, 0, 0, 0.5);
}
