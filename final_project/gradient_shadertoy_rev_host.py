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
    with open('loma_code/gradient_shadertoy_rev.py') as f:
        structs, lib = compiler.compile(f.read(),
                                  target = 'c',
                                  output_filename = '_code/gradient_shadertoy_rev')
    w = 100
    h = 100
    grad_f = lib.diff_shadertoy
    step_size = 1e-2
    d_vec3 = structs["Vec3"]
    d_vec4 = structs["Vec4"]
    d_float = structs["_dfloat"]

    cur_col1 = d_vec3(1, 0.6, 0.1)
    cur_col2 = d_vec3(0.5, 0.5, 0.5)
    target_col1 = d_vec3(1, 0.5, 0.5)
    target_col2= d_vec3(0, .2, 0.1)
    losses = []
    col1_x = []
    col1_y = []
    col1_z = []
    col2_x = []
    col2_y = []
    col2_z = []
    images = []

    epoch = 1000
    for i in range(epoch):
        loss = ctypes.c_float(0.0)
        img = np.zeros([h, w, 3], dtype = np.single)
        gradient = grad_f(w, h, cur_col1, cur_col2, target_col1, target_col2, loss, img.ctypes.data_as(ctypes.POINTER(structs['Vec3'])))
        #print("gradient is ", gradient.x, gradient.y, gradient.z)

        cur_col1.x -= step_size * gradient.x1
        cur_col1.y -= step_size * gradient.y1
        cur_col1.z -= step_size * gradient.z1

        cur_col2.x -= step_size * gradient.x2
        cur_col2.y -= step_size * gradient.y2
        cur_col2.z -= step_size * gradient.z2


        col1_x.append(cur_col1.x)
        col1_y.append(cur_col1.y)
        col1_z.append(cur_col1.z)
        col2_x.append(cur_col2.x)
        col2_y.append(cur_col2.y)
        col2_z.append(cur_col2.z)
        
        losses.append(loss.value)
        images.append(img)

    iterations = list(range(epoch))
    plt.figure(figsize=(10, 6))
    plt.title('Color Over Iterations')
    plt.plot(iterations, col1_x, label='col1_x', color='red', linewidth=2)
    plt.plot(iterations, col1_y, label='col1_y', color='green', linewidth=2)
    plt.plot(iterations, col1_z, label='col1_z', color='blue', linewidth=2)
    plt.plot(iterations, col2_x, label='col2_x', color='magenta', linewidth=2)
    plt.plot(iterations, col2_y, label='col2_y', color='yellow', linewidth=2)
    plt.plot(iterations, col2_z, label='col2_z', color='cyan', linewidth=2)
    plt.ylabel('Color')
   
    plt.xlabel('Iterations')
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

    # plt.imshow(images[0])
    # plt.show()
    # plt.imshow(images[epoch-1])
    # plt.show()
    # plt.imshow(images[1000])
    # plt.show()

