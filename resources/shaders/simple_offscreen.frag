#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "common.h"

layout(location = 0) out vec4 gAlbedo;
layout(location = 1) out vec4 gNormal;

layout(binding = 0, set = 0) uniform AppData
{
  UniformParams Params;
};

layout (location = 0) in VS_OUT
{
  vec3 wPos;
  vec3 wNorm;
} surf;

void main()
{
    gAlbedo = vec4(Params.baseColor, 1.0);
    gNormal = vec4(surf.wNorm, 1.0);
}