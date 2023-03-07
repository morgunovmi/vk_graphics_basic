#version 450

layout (set = 0, binding = 0) uniform sampler2D heightMap;

layout(quads, equal_spacing, cw) in;

layout (location = 0) in vec2 texCoord[];

layout (location = 0) out TSE_OUT
{
    vec3 wPos;
    vec3 wNorm;
    vec2 texCoord;
} teseOut;

layout(push_constant) uniform params_t
{
    mat4 mProjView;
    mat4 mModel;
} params;

void main()
{
    // Interpolate UV coordinates
	vec2 uv1 = mix(texCoord[0], texCoord[1], gl_TessCoord.x);
	vec2 uv2 = mix(texCoord[3], texCoord[2], gl_TessCoord.x);
	teseOut.texCoord = mix(uv1, uv2, gl_TessCoord.y);

    float height = textureLod(heightMap, teseOut.texCoord, 0.0).r;

    vec4 p00 = gl_in[0].gl_Position;
    vec4 p01 = gl_in[1].gl_Position;
    vec4 p10 = gl_in[2].gl_Position;
    vec4 p11 = gl_in[3].gl_Position;

    vec4 uVec = p01 - p00;
    vec4 vVec = p10 - p00;
    vec4 normal = normalize(vec4(cross(vVec.xyz, uVec.xyz), 0));
    teseOut.wNorm = normalize(mat3(transpose(inverse(params.mModel))) * normal.xyz);

	// Interpolate positions
	vec4 pos1 = mix(p00, p01, gl_TessCoord.x);
	vec4 pos2 = mix(p11, p10, gl_TessCoord.x);
	vec4 pos = mix(pos1, pos2, gl_TessCoord.y);

    pos += normal * height;

    teseOut.wPos = (params.mModel * pos).xyz;
    gl_Position = params.mProjView * vec4(teseOut.wPos, 1.0f);
}