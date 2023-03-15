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
	vec2 uv2 = mix(texCoord[2], texCoord[3], gl_TessCoord.x);
	teseOut.texCoord = mix(uv1, uv2, gl_TessCoord.y);

    float height = textureLod(heightMap, teseOut.texCoord, 0.0).r;
    float height_x_offset = textureLod(heightMap, vec2(teseOut.texCoord.x + 0.005, teseOut.texCoord.y), 0.0).r;
    float height_y_offset = textureLod(heightMap, vec2(teseOut.texCoord.x, teseOut.texCoord.y + 0.005), 0.0).r;

	// Interpolate positions
	vec4 pos1 = mix(gl_in[0].gl_Position, gl_in[1].gl_Position, gl_TessCoord.x);
	vec4 pos2 = mix(gl_in[2].gl_Position, gl_in[3].gl_Position, gl_TessCoord.x);
	vec3 pos = mix(pos1, pos2, gl_TessCoord.y).xyz;
    pos.z = height;
    teseOut.wPos = (params.mModel * vec4(pos, 1.0)).xyz;

    vec3 pos_x_offset = vec3(pos.x + 0.01, pos.y, height_x_offset);
    vec3 pos_y_offset = vec3(pos.x, pos.y + 0.01, height_y_offset);
    vec3 uVec = pos_x_offset - pos;
    vec3 vVec = pos_y_offset - pos;
    vec3 normal = normalize(cross(uVec, vVec));

    teseOut.wNorm = normalize(mat3(transpose(inverse(params.mModel))) * normal);

    gl_Position = params.mProjView * vec4(teseOut.wPos, 1.0f);
}