import os
import subprocess
import pathlib

if __name__ == '__main__':
    glslang_cmd = "glslangValidator"

    shader_list = ["simple.vert", "quad.vert", "quad.frag", "quad3_vert.vert",
         "simple_shadow.frag", "simple_geometry.frag", "simple_quad.vert", "simple_deferred.frag",
         "simple_ssao.frag", "gaussian_horizontal.comp", "gaussian_vertical.comp"]

    for shader in shader_list:
        subprocess.run([glslang_cmd, "-V", shader, "-o", "{}.spv".format(shader)])

