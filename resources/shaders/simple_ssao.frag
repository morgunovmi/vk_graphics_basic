#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) out float color;

layout (binding = 0) uniform Kernel
{
    vec3 samples[];
};

layout(push_constant) uniform params_t
{
    mat4 projInverse;
    mat4 viewInverse;
} params;

layout (binding = 1) uniform sampler2D texNoise;
layout (binding = 2) uniform sampler2D gNormal;
layout (binding = 3) uniform sampler2D depth;

layout (location = 0) in VS_OUT
{
  vec2 texCoord;
} vOut;

const vec2 noiseScale = vec2(1024 / 4.0, 1024 / 4.0);

void main()
{
    color =  textureLod(texNoise, vOut.texCoord * noiseScale, 0).x;
}
