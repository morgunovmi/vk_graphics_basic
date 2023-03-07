#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "unpack_attributes.h"

layout(push_constant) uniform params_t
{
    mat4 mProjView;
    mat4 mModel;
} params;

layout (location = 0 ) out VS_OUT
{
    vec3 wPos;
    vec3 wNorm;
    vec2 texCoord;
} vOut;

out gl_PerVertex { vec4 gl_Position; };

void main(void)
{
    vOut.texCoord = vec2(gl_VertexIndex / 2, int((gl_VertexIndex % 3) > 0));
    vec3 pos = vec3(vOut.texCoord * 2.0f - 1.0f, 0.0f);
    vOut.wPos  = (params.mModel * vec4(pos, 1.0f)).xyz;

    vec3 normal = vec3(0, 0, -1);
    vOut.wNorm = normalize(mat3(transpose(inverse(params.mModel))) * normal.xyz);
    gl_Position   = params.mProjView * vec4(vOut.wPos, 1.0);
}
