#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "common.h"

layout(location = 0) out vec4 out_fragColor;

layout (location = 0) in VS_OUT
{
  vec2 texCoord;
} vOut;

layout(binding = 0, set = 0) uniform AppData
{
    UniformParams Params;
};

layout(binding = 1) uniform sampler2D hdrImage;

void main() {
  out_fragColor = texture(hdrImage, vOut.texCoord);
}