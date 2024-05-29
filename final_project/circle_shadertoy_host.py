import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import compiler
import ctypes
import numpy as np
import matplotlib.pyplot as plt

def draw_circle(w, h, center_x, center_y, radius):
    circle_color = [0, 0, 1] # blue
    img = np.ones((w, h, 3)) # white background
    y, x = np.ogrid[:w, :h]
    distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    mask = (distance_from_center <= radius)
    img[mask] = circle_color 
    return img

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
    d_vec4 = structs["Vec4"]
    d_float = structs["_dfloat"]
    cur_radius = 0.2
    target_radius = 0.5
    cur_center = d_vec3(0, 0, -1)
    target_center = d_vec3(0, 0, -1)

    losses = []
    radii = []
    center_x = []
    center_y = []
    center_z = []

    epoch = 2000
    for i in range(epoch):
        loss = ctypes.c_float(0.0)
        gradient = grad_f(w, h, cur_radius, target_radius, cur_center, target_center,loss)
        #cur_center.x -= step_size * gradient.x 
        #cur_center.y -= step_size * gradient.y
        #cur_center.z -= step_size * gradient.z
        cur_radius -= step_size * gradient.w

        radii.append(cur_radius)
        #center_x.append(cur_center.x)
        #center_y.append(cur_center.y)
        #center_z.append(cur_center.z)
        #print("cur_radius at iteration " + str(i) + " is " + str(cur_radius))
        #print("cur_center is (" + str(cur_center.x) + ", " + str(cur_center.y) + ", " + str(cur_center.z) + ")")
        #print("gradient is " + str(gradient.x) + ", " + str(gradient.y) + ", " + str(gradient.z))
        losses.append(loss.value)

    iterations = list(range(epoch))
    plt.figure(figsize=(10, 6))
    #plt.plot(iterations, center_x, label='center x', color='blue', linewidth=2)
    #plt.plot(iterations, center_y, label='center y', color='red', linewidth=2)
    #plt.plot(iterations, center_z, label='center z', color='green', linewidth=2)
    plt.plot(iterations, radii, label='radius', color='red', linewidth=2)
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

    # w = 100
    # h = 100
    # cur_img = draw_circle(w, h, center_x=50, center_y=50, radius=10)
    # target_img = draw_circle(w, h, center_x=50, center_y=50, radius=20)
    # grad_f = lib.diff_shadertoy
    # step_size = 1e-2
    # d_vec3 = structs["Vec3"]
    # fig_row = 6
    # fig_col = 5
    # fig, axes = plt.subplots(fig_row, fig_col, figsize=(8, 8))
    # axes[fig_row-1, 0].imshow(cur_img)
    # axes[fig_row-1, 0].axis("off")
    # axes[fig_row-1, 0].set_title("Start image")
    # axes[fig_row-1, 1].imshow(target_img)
    # axes[fig_row-1, 1].axis("off")
    # axes[fig_row-1, 1].set_title("Target image")

    # for i in range(1000):
    #     loss = np.zeros([h, w, 3], dtype = np.single)
    #     grad_f(w, h, 
    #            cur_img.ctypes.data_as(ctypes.POINTER(structs['Vec3'])),
    #            target_img.ctypes.data_as(ctypes.POINTER(structs['Vec3'])), 
    #            loss.ctypes.data_as(ctypes.POINTER(structs['Vec3'])))
    #     cur_img -= step_size * loss
    #     if i % 40 == 0:
    #         j = i // 40
    #         axes[j // fig_col, j % fig_col].imshow(cur_img)
    #         axes[j // fig_col, j % fig_col].axis("off")
    #         axes[j // fig_col, j % fig_col].set_title("i = " + str(i))
    
    # axes[fig_row-1, 2].imshow(cur_img)
    # axes[fig_row-1, 2].axis("off")
    # axes[fig_row-1, 2].set_title("1000 iterations")
    # plt.subplots_adjust(wspace=0.4, hspace=0.4)
    # plt.show()


