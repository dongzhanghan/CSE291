class Vec3:
    x : float
    y : float
    z : float

def make_vec3(x : In[float], y : In[float], z : In[float]) -> Vec3:
    ret : Vec3
    ret.x = x
    ret.y = y
    ret.z = z
    return ret

def shadertoy(img : In[Array[Vec3]]) -> Array[Vec3]:
    return img

d_shadertoy = fwd_diff(shadertoy)


def diff_shadertoy(image:In[Array[Vec3]], w : In[int], h : In[int], diff_image : Out[Array[Vec3]]):
    y : int = 0
    x : int
    d_image : Array[Diff[Vec3],10000]
    d_color : Array[Diff[Vec3],10000]
    while (y < h, max_iter := 4096):
        x = 0
        while (x < w, max_iter := 4096):
            d_image[w * y + x].val = image.val
            d_image[w * y + x].dval = 1
            x = x + 1
        y = y + 1
    y = 0
    while (y < h, max_iter := 4096):
        x = 0
        while (x < w, max_iter := 4096):
            d_color = d_shadertoy(d_image)
            diff_image[w * y + x].x = d_color[w * y + x].x.dval*2*(image[w * y + x].x-0.4)
            diff_image[w * y + x].y = d_color[w * y + x].y.dval*2*(image[w * y + x].y-0.8)
            diff_image[w * y + x].z = d_color[w * y + x].z.dval*2*(image[w * y + x].z-0.6)
            x = x + 1
        y = y + 1



