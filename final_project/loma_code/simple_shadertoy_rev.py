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

d_shadertoy = rev_diff(shadertoy)


def diff_shadertoy(col: In[Vec3], w : In[int], h : In[int], loss:Out[Vec3])->Vec3:
    y : int = 0
    x : int
    gradient: Vec3
    d_col:Vec3
    while (y < h, max_iter := 4096):
        x = 0
        while (x < w, max_iter := 4096):
            d_col = make_vec3(0,0,0)
            d_shadertoy(col,d_col,make_vec3(1.0,1.0,1.0))
            gradient.x = gradient.x+d_col.x*2*(col.x-0.4)
            loss.x = loss.x+ (col.x-0.4)*(col.x-0.4)
            gradient.y = gradient.y+d_col.y*2*(col.y-0.8)
            loss.y = loss.y+ (col.y-0.8)*(col.y-0.8)
            gradient.z = gradient.z+d_col.z*2*(col.z-0.7)
            loss.z = loss.z+ (col.z-0.7)*(col.z-0.7)
            x = x + 1
        y = y + 1
    gradient.x = gradient.x/(w*h)
    gradient.y = gradient.y/(w*h)
    gradient.z = gradient.z/(w*h)
    return gradient



