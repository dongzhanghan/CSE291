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

def shadertoy(col: In[Vec3])-> Vec3:
    return col

d_shadertoy = fwd_diff(shadertoy)


def diff_shadertoy(w : In[int], h : In[int], cur_img : In[Array[Vec3]], target_img : In[Array[Vec3]], loss : Out[Array[Vec3]]):
    y : int = 0
    x : int
    d_color : Diff[Vec3]
    d_col: Diff[Vec3]
    while (y < h, max_iter := 4096):
        x = 0
        while (x < w, max_iter := 4096):
            d_col.x.val = cur_img[w * y + x].x
            d_col.y.val = cur_img[w * y + x].y
            d_col.z.val = cur_img[w * y + x].z
            d_col.x.dval = 1
            d_col.y.dval = 1
            d_col.z.dval = 1
            d_color = d_shadertoy(d_col)
            loss[w * y + x].x = loss[w * y + x].x+d_color.x.dval*2*(d_color.x.val-target_img[w * y + x].x)
            loss[w * y + x].y = loss[w * y + x].y+d_color.y.dval*2*(d_color.y.val-target_img[w * y + x].y)
            loss[w * y + x].z = loss[w * y + x].z+d_color.z.dval*2*(d_color.z.val-target_img[w * y + x].z)
            x = x + 1
        y = y + 1


