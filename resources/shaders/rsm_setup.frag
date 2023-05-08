#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "common.h"

layout(location = 0) out vec4 wPos;
layout(location = 1) out vec4 wNormal;
layout(location = 2) out vec4 flux;

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

const vec4 lightColor = vec4(1.0, 1.0, 1.0, 1.0);

void main()
{
    wPos = vec4(surf.wPos, 1.0);
    wNormal = vec4(surf.wNorm, 1.0);
    flux = params.objColor * lightColor;
}