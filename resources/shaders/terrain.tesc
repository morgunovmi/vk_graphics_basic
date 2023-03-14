#version 450

layout (vertices = 4) out;

layout (location = 0) in vec2 texCoord[];
layout (location = 0) out vec2 textureCoords[];

void main()
{
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    textureCoords[gl_InvocationID] = texCoord[gl_InvocationID];

    if (gl_InvocationID == 0)
	{
        gl_TessLevelOuter[0] = 1;
        gl_TessLevelOuter[1] = 1;
        gl_TessLevelOuter[2] = 1;
        gl_TessLevelOuter[3] = 1;

        gl_TessLevelInner[0] = 1;
        gl_TessLevelInner[1] = 1;
    }
}