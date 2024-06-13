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


def diff_shadertoy(col: In[Vec3], w : In[int], h : In[int], loss:Out[Vec3])->Vec3:
    y : int = 0
    x : int
    d_color : Diff[Vec3]
    d_col: Diff[Vec3]
    gradient: Vec3
    while (y < h, max_iter := 4096):
        x = 0
        while (x < w, max_iter := 4096):
            d_col.x.val = col.x
            d_col.y.val = col.y
            d_col.z.val = col.z
            d_col.x.dval = 1
            d_col.y.dval = 1
            d_col.z.dval = 1
            d_color = d_shadertoy(d_col)
            gradient.x = gradient.x+d_color.x.dval*2*(d_color.x.val-0.4)
            loss.x = loss.x+ (d_color.x.val-0.4)*(d_color.x.val-0.4)
            gradient.y = gradient.y+d_color.y.dval*2*(d_color.y.val-0.8)
            loss.y = loss.y+ (d_color.y.val-0.8)*(d_color.y.val-0.8)
            gradient.z = gradient.z+d_color.z.dval*2*(d_color.z.val-0.7)
            loss.z = loss.z+ (d_color.z.val-0.7)*(d_color.z.val-0.7)
            x = x + 1
        y = y + 1
    gradient.x = gradient.x/(w*h)
    gradient.y = gradient.y/(w*h)
    gradient.z = gradient.z/(w*h)
    return gradient



