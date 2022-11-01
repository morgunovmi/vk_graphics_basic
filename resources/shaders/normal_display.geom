#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "common.h"

layout (triangles) in;
layout (line_strip, max_vertices = 6) out;

layout(push_constant) uniform params_t
{
    mat4 mProjView;
    mat4 mModel;
} params;

layout(binding = 0, set = 0) uniform AppData
{
    UniformParams Params;
};

layout (location = 0) in VS_OUT
{
    vec3 wPos;
    vec3 wNorm;
} gs_in[];

const float MAGNITUDE = 0.04;

void generate_line(uint index)
{
    gl_Position = params.mProjView * vec4(gs_in[index].wPos, 1.0);
    EmitVertex();
    gl_Position = params.mProjView * vec4(gs_in[index].wPos + 
                                     gs_in[index].wNorm * MAGNITUDE * (1.0 + sin(Params.time)), 1.0);
    EmitVertex();
    EndPrimitive();
}

void main()
{
    generate_line(0);
    generate_line(1);
    generate_line(2);
}  