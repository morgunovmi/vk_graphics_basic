import os
import subprocess
import pathlib

if __name__ == '__main__':
    glslang_cmd = "glslangValidator"

    shader_list = ["simple.vert", "quad.vert", "quad.frag",
                   "simple_shadow.frag", "noise_quad.vert",
                   "noise.frag", "terrain.vert", "terrain.frag",
                   "terrain.tesc", "terrain.tese",
                   "fog.vert", "fog.frag", "quad3_vert.vert",
                   "fog_display.frag"]

    for shader in shader_list:
        subprocess.run([glslang_cmd, "-V", shader, "-o", "{}.spv".format(shader)])

