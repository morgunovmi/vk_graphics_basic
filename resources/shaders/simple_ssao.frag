#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "common.h"

layout(location = 0) out float color;

layout (binding = 0) uniform Kernel
{
    vec4 samples[64];
};

layout(binding = 1) uniform AppData
{
  UniformParams ubo;
};

layout(push_constant) uniform params_t
{
    mat4 projInverse;
    mat4 viewInverse;
} params;

layout (binding = 2) uniform sampler2D texNoise;
layout (binding = 3) uniform sampler2D gNormal;
layout (binding = 4) uniform sampler2D gPosition;

layout (location = 0) in VS_OUT
{
  vec2 texCoord;
} vOut;

const vec2 noiseScale = vec2(1024 / 4.0, 1024 / 4.0);

const int kernelSize = 64;
const float radius = 0.5;
const float bias = 0.025;

void main()
{
    const vec4 vPos = textureLod(gPosition, vOut.texCoord, 0);
    const vec3 vNorm = normalize(textureLod(gNormal, vOut.texCoord, 0).xyz);
    const vec3 randomVec = normalize(texture(texNoise, vOut.texCoord * noiseScale).xyz);

    const vec3 tangent = normalize(randomVec - vNorm * dot(randomVec, vNorm));
    const vec3 bitangent = cross(vNorm, tangent);
    mat3 TBN = mat3(tangent, bitangent, vNorm);

    float occlusion = 0.0;
    for (int i = 0; i < kernelSize; ++i)
    {
        vec3 samplePos = TBN * samples[i].xyz;
        samplePos = vPos.xyz + samplePos * radius;

        vec4 offset = vec4(samplePos, 1.0);
        offset = ubo.projMat * offset;
        offset.xyz /= offset.w;
        offset.xyz = offset.xyz * 0.5 + 0.5;

        float sampleDepth = texture(gPosition, offset.xy).z;

        occlusion += (sampleDepth >= samplePos.z + bias ? 1.0 : 0.0);
    }
    
    color = 1.0 - (occlusion / kernelSize);
}
