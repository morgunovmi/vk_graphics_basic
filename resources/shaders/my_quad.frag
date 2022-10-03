#version 450
#extension GL_ARB_separate_shader_objects : enable

layout (location = 0) out vec4 color;

layout (binding = 0) uniform sampler2D colorTex;

layout (location = 0) in VS_OUT
{
  vec2 texCoord;
} surf;

float luminance(vec4 color)
{
  return length(color.xyz);
}

float sigmaD = 0.5;
float sigmaR = 0.1;
int rad = 2;

void main()
{
  float mulD = -1.0 / (2.0 * sigmaD * sigmaD);
  float mulR = -1.0 / (2.0 * sigmaR * sigmaR);

  float sumWeight = 0.0;
  vec4 sumColor = vec4(0.0);

  ivec2 texSize = textureSize(colorTex, 0);
  float centerL = luminance(texture(colorTex, surf.texCoord));

  for (int i = -rad; i < rad; ++i)
  {
    for (int j = -rad; j < rad; ++j)
    {
      vec2 offset = vec2(i, j);
      vec4 curColor = texture(colorTex, surf.texCoord + offset / texSize);

      float distD = length(offset);
      float distR = luminance(curColor) - centerL;

      float w = exp(mulD * distD * distD) * exp(mulR * distR * distR);

      sumColor += w * curColor;
      sumWeight += w;
    }
  }

  color = sumColor / sumWeight;
}