#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) out float color;

layout (binding = 0) uniform Kernel
{
    vec4 samples[];
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
    float x = vOut.texCoord.x * 2.0 - 1.0;
    float y = vOut.texCoord.y * 2.0 - 1.0;
    float z = textureLod(depth, vOut.texCoord, 0).x;

    vec4 clipSpacePosition = vec4(x, y, z, 1.0);
    vec4 viewSpacePosition = params.projInverse * clipSpacePosition;
    viewSpacePosition /= viewSpacePosition.w;
    vec4 wPos = params.viewInverse * viewSpacePosition;

    vec4 meme = samples[63];
    
    color =  textureLod(texNoise, vOut.texCoord * noiseScale, 0).x;
}
