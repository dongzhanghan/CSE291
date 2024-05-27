import matplotlib.pyplot as plt
import numpy as np

w = 100
h = 100
center_x, center_y = 50, 50 
radius = 20
circle_color = [0, 0, 1] # blue

img = np.ones((w, h, 3)) # white background
y, x = np.ogrid[:w, :h]
distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
mask = (distance_from_center <= radius)
img[mask] = circle_color 

plt.imshow(img)
plt.show()