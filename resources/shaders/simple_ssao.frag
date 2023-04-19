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
layout (binding = 4) uniform sampler2D depth;

layout (location = 0) in VS_OUT
{
  vec2 texCoord;
} vOut;

const vec2 noiseScale = vec2(1024 / 4.0, 1024 / 4.0);
const float radius = 0.5;
const float bias = 0.025;

void main()
{
    float x = vOut.texCoord.x * 2.0 - 1.0;
    float y = vOut.texCoord.y * 2.0 - 1.0;
    float z = textureLod(depth, vOut.texCoord, 0).x;

    vec4 clipSpacePosition = vec4(x, y, z, 1.0);
    vec4 viewSpacePosition = params.projInverse * clipSpacePosition;
    viewSpacePosition /= viewSpacePosition.w;

    vec3 wNorm = texture(gNormal, vOut.texCoord).rgb;
    vec3 vNorm = (ubo.viewMat * vec4(wNorm, 1.0)).xyz;
    vec3 randomVec = texture(texNoise, vOut.texCoord * noiseScale).xyz;

    vec3 tangent = normalize(randomVec - vNorm * dot(randomVec, vNorm));
    vec3 bitangent = cross(vNorm, tangent);
    mat3 TBN = mat3(tangent, bitangent, vNorm);

    float occlusion = 0.0;
    for (int i = 0; i < 64; ++i)
    {
        vec3 samplePos = TBN * samples[i].xyz;
        samplePos = viewSpacePosition.xyz + samplePos * radius;

        vec4 offset = vec4(samplePos, 1.0);
        offset = ubo.projMat * offset;
        offset.xyz /= offset.w;
        offset.xyz = offset.xyz * 0.5 + 0.5;

        float sampleDepth = textureLod(depth, offset.xy, 0).x;

        occlusion += (sampleDepth >= samplePos.z + bias ? 1.0 : 0.0);
    }
    
    color =  occlusion;
}
