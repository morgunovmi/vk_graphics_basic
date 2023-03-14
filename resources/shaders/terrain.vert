#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "unpack_attributes.h"

layout (location = 0) out vec2 texCoord;

void main(void)
{
    texCoord = vec2(int((gl_VertexIndex % 3) > 0), gl_VertexIndex / 2);
    vec3 pos = vec3(texCoord * 2.0f - 1.0f, 0.0f);

    gl_Position   = vec4(pos, 1.0);
}
