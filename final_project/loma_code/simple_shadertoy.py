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

def shadertoy(col : In[Vec3]) -> Vec3:
    return col

d_shadertoy = fwd_diff(shadertoy)


def diff_shadertoy(w : In[int], h : In[int], image : Out[Array[Vec3]]):
    y : int = 0
    x : int
    col : Diff[Vec3] 
    col.x.val = 0.5
    col.y.val = 0.5
    col.z.val = 0.5
    d_color : Diff[Vec3]
    while (y < h, max_iter := 4096):
        x = 0
        while (x < w, max_iter := 4096):
            d_color = d_shadertoy(col)
            image[w * y + x].x = d_color.x.dval
            image[w * y + x].y = d_color.y.dval
            image[w * y + x].z = d_color.z.dval
            x = x + 1
        y = y + 1
