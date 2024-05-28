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
    img = np.ones((w, h, 3))
    grad_f = lib.diff_shadertoy
    step_size = 1e-2
    d_vec3 = structs["Vec3"]

    cur_radius = 0.5
    target_radius = 0.5
    cur_center = d_vec3(0, 0, -1)
    target_center = d_vec3(0, 1, -1)

    loss_r = []
    loss_g = []
    loss_b = []
    radii = []
    center_x = []
    center_y = []
    center_z = []

    fig_row = 10
    fig_col = 5
    fig, axes = plt.subplots(fig_row, fig_col, figsize=(8, 8))
    for i in range(1000):
        loss = d_vec3(0,0,0)
        gradient = grad_f(w, h, cur_radius, target_radius, cur_center, target_center, img.ctypes.data_as(ctypes.POINTER(structs['Vec3'])), loss)
        #cur_radius -= step_size * (gradient.x + gradient.y + gradient.z) / 3
        cur_center.x -= step_size * gradient.x #color r vs coordinate x?
        cur_center.y -= step_size * gradient.y
        loss_r.append(loss.x)
        loss_g.append(loss.y)
        loss_b.append(loss.z)
        #radii.append(cur_radius)
        center_x.append(cur_center.x)
        center_y.append(cur_center.y)
        if (i < 10 or i % 100 == 0):
            print("cur_radius at iteration " + str(i) + " is " + str(cur_radius))
            print("cur_center is (" + str(cur_center.x) + ", " + str(cur_center.y) + ", " + str(cur_center.z) + ")")
            print("gradient is " + str(gradient.x) + ", " + str(gradient.y) + ", " + str(gradient.z))
            print("loss is " + str(loss.x) + ", " + str(loss.y) + ", " + str(loss.z))
        if i % 10 == 0 and i < 500:
            j = i // 10
            axes[j // fig_col, j % fig_col].imshow(img)
            axes[j // fig_col, j % fig_col].axis("off")
            axes[j // fig_col, j % fig_col].set_title("i = " + str(i))
    #plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show()

    iterations = list(range(1000))

    # Plot center
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, center_x, label='center x', color='red', linewidth=2)
    plt.plot(iterations, center_y, label='center y', color='red', linewidth=2)
    plt.title('Change in Center Over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Center Coordinates')
    plt.legend()

    plt.grid(True)
    plt.show()

    # # Plot radius
    # plt.figure(figsize=(10, 6))
    # plt.plot(iterations, radii, label='radius', color='red', linewidth=2)
    # plt.title('Change in Radius Over Iterations')
    # plt.xlabel('Iterations')
    # plt.ylabel('Radius')
    # plt.legend()

    # plt.grid(True)
    # plt.show()

    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, loss_r, label='R Loss', color='red', linewidth=2)
    plt.plot(iterations, loss_g, label='G Loss', color='green', linewidth=2)
    plt.plot(iterations, loss_b, label='B Loss', color='blue', linewidth=2)
    plt.title('Loss Function Over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Loss Value')
    plt.legend()

    plt.grid(True)
    plt.show()

