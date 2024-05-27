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
    cur_img = draw_circle(w, h, center_x=50, center_y=50, radius=10)
    target_img = draw_circle(w, h, center_x=50, center_y=50, radius=20)
    grad_f = lib.diff_shadertoy
    step_size = 1e-2
    d_vec3 = structs["Vec3"]

    fig_row = 6
    fig_col = 5
    fig, axes = plt.subplots(fig_row, fig_col, figsize=(8, 8))
    axes[fig_row-1, 0].imshow(cur_img)
    axes[fig_row-1, 0].axis("off")
    axes[fig_row-1, 0].set_title("Start image")
    axes[fig_row-1, 1].imshow(target_img)
    axes[fig_row-1, 1].axis("off")
    axes[fig_row-1, 1].set_title("Target image")

    for i in range(25): #range(1000):
        loss = np.zeros([h, w, 3], dtype = np.single)
        grad_f(w, h, 
               cur_img.ctypes.data_as(ctypes.POINTER(structs['Vec3'])),
               target_img.ctypes.data_as(ctypes.POINTER(structs['Vec3'])), 
               loss.ctypes.data_as(ctypes.POINTER(structs['Vec3'])))
        cur_img -= step_size * loss
        axes[i // fig_col, i % fig_col].imshow(cur_img)
        axes[i // fig_col, i % fig_col].axis("off")
        axes[i // fig_col, i % fig_col].set_title("i = " + str(i))
    
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show()


