import os
import subprocess
import pathlib

if __name__ == '__main__':
    glslang_cmd = "glslangValidator"

    shader_list = ["simple.vert", "quad.vert", "quad.frag",
         "simple_shadow.frag", "simple_geometry.frag", "simple_quad.vert", "simple_deferred.frag",
         "rsm_setup.frag", "indirect_light.frag", "tonemap.frag"]

    for shader in shader_list:
        subprocess.run([glslang_cmd, "-V", shader, "-o", "{}.spv".format(shader)])

