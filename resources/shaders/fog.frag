#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) out vec4 color;

layout (location = 0 ) in VS_OUT
{
    vec3 wPos;
    vec3 oPos;
} surf;

layout(push_constant) uniform params_t
{
    mat4 mProjView;
    mat4 mModel;
    vec3 wCameraPos;
} params;

vec2 boxIntersection(vec3 ro, vec3 rd, vec3 boxSize)
{
    vec3 m = 1.0 / rd;
    vec3 n = m * ro;
    vec3 k = abs(m) * boxSize;
    vec3 t1 = -n - k;
    vec3 t2 = -n + k;
    float tN = max(max(t1.x, t1.y), t1.z);
    float tF = min(min(t2.x, t2.y), t2.z);
    if(tN > tF || tF < 0.0)
    {
    return vec2(-1.0, -1.0);
    }
    return vec2(tN, tF);
}

void main()
{
    vec3 worldPosToEye = normalize(params.wCameraPos - surf.wPos);
    vec3 rd = (inverse(params.mModel) * vec4(worldPosToEye, 0.0)).xyz;
    vec3 ro = surf.oPos - 1.0 * rd;

    vec3 boxSize = vec3(1.0, 1.0, 1.0);
    vec2 hits = boxIntersection(ro, rd, boxSize);

    vec3 exit = ro + hits.x * rd;
    vec3 entry = ro + hits.y * rd;

    vec3 segmentCenter = entry + (exit - entry) / 2.0;

    if (hits.x < 0)
        discard;

    color = vec4(length(segmentCenter.xyz), 0, 0, 1);
}