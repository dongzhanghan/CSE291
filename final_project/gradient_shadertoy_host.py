import os
import cv2
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import compiler
import ctypes
import numpy as np
import matplotlib.pyplot as plt



# def create_gradient_image(col1, col2):
#     img = np.ones((100, 100, 3))
#     y, x = np.ogrid[:100, :100]
#     img = (1-x/100) * col1 + x/100 * col2
#     return img

if __name__ == '__main__':
    with open('loma_code/gradient_shadertoy.py') as f:
        structs, lib = compiler.compile(f.read(),
                                  target = 'c',
                                  output_filename = '_code/gradient_shadertoy')
    w = 100
    h = 100
    grad_f = lib.diff_shadertoy
    step_size = 1e-2
    d_vec3 = structs["Vec3"]
    d_vec4 = structs["Vec4"]
    d_float = structs["_dfloat"]

    cur_col1 = d_vec3(0, 0.5, 0.1)
    cur_col2 = d_vec3(1, 1, 1)
    target_col1 = d_vec3(0.5, 0.5, 0.5)
    target_col2= d_vec3(0.8, 0.2, 0.7)
    losses = []
    col1_x = []
    col1_y = []
    col1_z = []
    col2_x = []
    col2_y = []
    col2_z = []
    images = []

    epoch = 2000
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

    plt.imshow(images[0])
    plt.show()
    plt.imshow(images[epoch-1])
    plt.show()
    plt.imshow(images[1000])
    plt.show()

    output_folder = 'output_images'
    os.makedirs(output_folder, exist_ok=True)
    for i in range(len(images)):
        img = (images[i] * 255).astype(np.uint8)

        # 生成文件名，例如 0001.png, 0002.png, ...
        file_name = f'{i+1:04d}.png'
        # 生成完整的文件路径
        file_path = os.path.join(output_folder, file_name)
        # 将图像保存为 PNG 文件
        cv2.imwrite(file_path, img)


