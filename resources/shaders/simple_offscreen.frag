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

float rand(vec2 co){
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

void main()
{
    gAlbedo = vec4(rand(vec2(surf.wNorm.x, surf.wNorm.y)), 0.1, 0.7, 1.0);
    gNormal = vec4(surf.wNorm, 1.0);
}