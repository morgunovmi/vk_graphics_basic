#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "unpack_attributes.h"

layout(location = 0) in vec4 vPosNorm;
layout(location = 1) in vec4 vTexCoordAndTang;

layout(push_constant) uniform params_t
{
    mat4 mProjView;
    vec4 modelRow1;
    vec4 modelRow2;
    vec4 modelRow3;
    vec4 objColor;
} params;

layout (location = 0 ) out VS_OUT
{
    vec3 wPos;
    vec3 wNorm;
} vOut;

out gl_PerVertex { vec4 gl_Position; };

mat4 collectModelMatrix(vec4 row1, vec4 row2, vec4 row3)
{
    mat4 mat;
    for (uint i = 0; i < 4; ++i)
    {
        mat[i][0] = row1[i];
        mat[i][1] = row2[i];
        mat[i][2] = row3[i];
    }
    mat[0][3] = 0;
    mat[1][3] = 0;
    mat[2][3] = 0;
    mat[3][3] = 1;
    return mat;
}

void main(void)
{
    mat4 mModel = collectModelMatrix(params.modelRow1, params.modelRow2, params.modelRow3);

    const vec4 wNorm = vec4(DecodeNormal(floatBitsToInt(vPosNorm.w)),         0.0f);

    vOut.wPos     = (mModel * vec4(vPosNorm.xyz, 1.0f)).xyz;
    vOut.wNorm    = normalize(mat3(transpose(inverse(mModel))) * wNorm.xyz);

    gl_Position   = params.mProjView * vec4(vOut.wPos, 1.0);
}
