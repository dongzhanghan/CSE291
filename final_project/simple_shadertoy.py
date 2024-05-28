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

    loss_r = []
    loss_g = []
    loss_b = []
    for i in range(1000):
        loss = d_vec3(0,0,0)
        gradient = grad_f(d_vec3(x,y,z),w, h, loss)
        x -= step_size * gradient.x
        y -= step_size * gradient.y
        z -= step_size * gradient.z
        loss_r.append(loss.x)
        loss_g.append(loss.y)
        loss_b.append(loss.z)
    iterations = list(range(1000))  
    
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
    print(x,y,z)


    
