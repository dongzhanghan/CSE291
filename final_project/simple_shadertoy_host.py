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
    w = 400
    h = 225
    loss = np.zeros([h, w, 3], dtype = np.single)
    img = np.ones([h, w, 3], dtype = np.single)
    x = 0.5
    y = 0.5
    z = 0.5
    grad_f = lib.diff_shadertoy
    step_size = 1e-2
    for i in range(100):
        grad_f(img.ctypes.data_as(ctypes.POINTER(structs['Vec3'])),w, h, loss.ctypes.data_as(ctypes.POINTER(structs['Vec3'])))
        for j in range(w):
            for k in range(h):
                img[0].val -= step_size * loss[0].val

    
