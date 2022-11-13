#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "common.h"

layout(location = 0) out vec4 gAlbedo;
layout(location = 1) out vec4 gNormal;

layout (location = 0) in VS_OUT
{
  vec3 wPos;
  vec3 wNorm;
} surf;

void main()
{
    gAlbedo = vec4(0.3, 0.3, 0.7, 1.0);
    gNormal = vec4(surf.wNorm, 1.0);
}