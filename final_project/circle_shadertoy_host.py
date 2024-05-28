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
    with open('loma_code/circle_shadertoy.py') as f:
        structs, lib = compiler.compile(f.read(),
                                  target = 'c',
                                  output_filename = '_code/circle_shadertoy')
    w = 100
    h = 100
    img_x = np.ones((w, h, 3))
    img_y = np.ones((w, h, 3))
    grad_f = lib.diff_shadertoy
    step_size = 1e-2
    d_vec3 = structs["Vec3"]
    d_float = structs["_dfloat"]
    cur_radius = 0.5
    target_radius = 0.5
    cur_center = d_vec3(0.2, 0.2, -1)
    target_center = d_vec3(0.3, 0.3, -1)

    losses = []
    radii = []
    center_x = []
    center_y = []
    center_z = []

    epoch = 2000
    for i in range(epoch):
        loss = ctypes.c_float(0.0)
        gradient = grad_f(w, h, cur_radius, target_radius, cur_center, target_center, img_x.ctypes.data_as(ctypes.POINTER(structs['Vec3'])), img_y.ctypes.data_as(ctypes.POINTER(structs['Vec3'])),loss)
        #cur_radius -= step_size * (gradient.x + gradient.y + gradient.z) / 3
        cur_center.x -= step_size * gradient.x #color r vs coordinate x?
        cur_center.y -= step_size * gradient.y
        losses.append(loss.value)
        #radii.append(cur_radius)
        center_x.append(cur_center.x)
        center_y.append(cur_center.y)
        #print("cur_radius at iteration " + str(i) + " is " + str(cur_radius))
        #print("cur_center is (" + str(cur_center.x) + ", " + str(cur_center.y) + ", " + str(cur_center.z) + ")")
        #print("gradient is " + str(gradient.x) + ", " + str(gradient.y) + ", " + str(gradient.z))

    iterations = list(range(epoch))
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, center_x, label='center x', color='blue', linewidth=2)
    plt.plot(iterations, center_y, label='center y', color='red', linewidth=2)
    # plt.plot(iterations, radii, label='radius', color='red', linewidth=2)
    plt.title('Center Over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Center')
    plt.legend()

    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, losses, label='Loss', color='red', linewidth=2)
    plt.title('Loss Function Over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Loss Value')
    plt.legend()

    plt.grid(True)
    plt.show()

