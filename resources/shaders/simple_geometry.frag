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

layout(push_constant) uniform params_t
{
    mat4 mProjView;
    vec4 modelRow1;
    vec4 modelRow2;
    vec4 modelRow3;
    vec4 objColor;
} params;

void main()
{
    gAlbedo = params.objColor;
    gNormal = vec4(surf.wNorm, 1.0);
}