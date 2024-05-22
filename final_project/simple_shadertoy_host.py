import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import compiler
import ctypes
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    with open('loma_code/simple_shadertoy.py') as f:
        structs, lib = compiler.compile(f.read(),
                                  target = 'c',
                                  output_filename = '_code/simple_shadertoy')
    w = 100
    h = 100
    x = 0.5
    y = 0.5
    z = 0.5
    grad_f = lib.diff_shadertoy
    step_size = 1e-2
    d_vec3 = structs["Vec3"]
    for i in range(1000):
        loss = grad_f(d_vec3(x,y,z),w, h)
        x -= step_size * loss.x
        y -= step_size * loss.y
        z -= step_size * loss.z



    
