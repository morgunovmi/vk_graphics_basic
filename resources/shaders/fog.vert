#version 450

layout(push_constant) uniform params_t
{
    mat4 mProjView;
    mat4 mModel;
    vec3 wCameraPos;
} params;

layout (location = 0 ) out VS_OUT
{
    vec3 wPos;
    vec3 oPos;
} vOut;

void main(void)
{
    vec3 pos = vec3(gl_VertexIndex / 4, gl_VertexIndex / 2 % 2 == 1, (gl_VertexIndex % 2) != 0);
    vOut.oPos = (pos - vec3(0.5, 0.5, 0.5)) * 2;

    vOut.wPos = (params.mModel * vec4(vOut.oPos, 1.0)).xyz;
    gl_Position   = params.mProjView * vec4(vOut.wPos, 1.0);
}