#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "common.h"

layout (location = 0) out VS_OUT
{
    vec2 texCoord;
} vOut;

layout(push_constant) uniform params_t
{
    mat4 mProjView;
    mat4 mModel;
} params;

layout(std430, binding = 0) buffer particleDrawListBuffer
{
    ParticleDrawData drawData[];
};

mat3 rotateZ(float alpha)
{
    float c = cos(alpha);
    float s = sin(alpha);
    return mat3(vec3(c, -s, 0),
                vec3(s, c, 0),
                vec3(0, 0, 1));
}

vec3 transform(uint id, vec3 pos)
{
    ParticleDrawData data = drawData[id];
    return data.pos.xyz + rotateZ(data.rot) * data.scale * pos;
}

void main() {
    vec2 xy = vec2(0);
    uint id = gl_InstanceIndex;
    if (gl_VertexIndex == 0 || gl_VertexIndex == 5) xy = vec2(-1,  1);
    if (gl_VertexIndex == 1)                        xy = vec2(-1, -1);
    if (gl_VertexIndex == 2 || gl_VertexIndex == 3) xy = vec2( 1, -1);
    if (gl_VertexIndex == 4)                        xy = vec2( 1,  1);
    vec3 localPos = vec3(xy, 0.0);
    vec3 emitterSpacePos = transform(id, localPos);

    gl_Position   = params.mProjView * params.mModel * vec4(emitterSpacePos, 1);
    vOut.texCoord = xy * 0.5 + 0.5;
}