#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "common.h"

layout(location = 0) out vec4 gPosition;
layout(location = 1) out vec4 gNormal;

layout (location = 0) in VS_OUT
{
  vec3 wPos;
  vec3 wNorm;
} surf;

void main()
{
    gPosition = vec4(surf.wPos, 1.0);
    gNormal = vec4(surf.wNorm, 1.0);
}