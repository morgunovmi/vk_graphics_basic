#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "common.h"

layout(triangles) in;
layout(triangle_strip, max_vertices = 6) out;

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
    vec3 wTangent;
    vec2 texCoord;
} gs_in[];

layout (location = 0) out GS_OUT
{
    vec3 wPos;
    vec3 wNorm;
    vec3 wTangent;
    vec2 texCoord;
} gs_out;

vec3 GetNormal()
{
   vec3 a = vec3(gs_in[0].wPos) - vec3(gs_in[1].wPos);
   vec3 b = vec3(gs_in[2].wPos) - vec3(gs_in[1].wPos);
   return normalize(cross(a, b));
}

vec3 explode(vec3 position, vec3 normal)
{
    float magnitude = 2.0;
    vec3 direction = normal * ((sin(Params.time) + 1.0) / 2.0) * magnitude; 
    return position + direction;
} 

void main()
{
    vec3 normal = GetNormal();

    for (uint i = 0; i < gl_in.length(); ++i)
    {
        gs_out.wPos = explode(gs_in[i].wPos, normal);
        gl_Position = params.mProjView * vec4(gs_out.wPos, 1.0);
        gs_out.wNorm = gs_in[i].wNorm;
        gs_out.wTangent = gs_in[i].wTangent;
        gs_out.texCoord = gs_in[i].texCoord;
        EmitVertex();
    }

    EndPrimitive();
}