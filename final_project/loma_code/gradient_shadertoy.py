class Vec3:
    x : float
    y : float
    z : float

class Vec4:
    x : float
    y : float
    z : float
    w : float

class Vec6:
    x1 : float
    y1 : float
    z1 : float
    x2 : float
    y2 : float
    z2 : float

class Sphere:
    center : Vec3
    radius : float

class Ray:
    org : Vec3
    dir : Vec3

def make_vec3(x : In[float], y : In[float], z : In[float]) -> Vec3:
    ret : Vec3
    ret.x = x
    ret.y = y
    ret.z = z
    return ret

def make_vec4(x : In[float], y : In[float], z : In[float], w: In[float]) -> Vec4:
    ret : Vec4
    ret.x = x
    ret.y = y
    ret.z = z
    ret.w = w
    return ret

def make_vec6(x1 : In[float], y1 : In[float], z1 : In[float],
              x2 : In[float], y2 : In[float], z2 : In[float]) -> Vec6:
    ret : Vec6
    ret.x1 = x1
    ret.y1 = y1
    ret.z1 = z1
    ret.x2 = x2
    ret.y2 = y2
    ret.z2 = z2
    return ret

def add(a : In[Vec3], b : In[Vec3]) -> Vec3:
    return make_vec3(a.x + b.x, a.y + b.y, a.z + b.z)

def sub(a : In[Vec3], b : In[Vec3]) -> Vec3:
    return make_vec3(a.x - b.x, a.y - b.y, a.z - b.z)

def mul(a : In[float], b : In[Vec3]) -> Vec3:
    return make_vec3(a * b.x, a * b.y, a * b.z)

def dot(a : In[Vec3], b : In[Vec3]) -> float:
    return a.x * b.x + a.y * b.y + a.z * b.z

def normalize(v : In[Vec3]) -> Vec3:
    l : float = sqrt(dot(v, v))
    return make_vec3(v.x / l, v.y / l, v.z / l)

def average(v:In[Vec3])->float:
    return (v.x+v.y+v.z)/3


def ray_color(xw_ratio : In[float], col1 : In[Vec3], col2 : In[Vec3]) -> Vec3:
    return add(mul((1-xw_ratio), col1), mul(xw_ratio, col2))

d_ray_color = fwd_diff(ray_color)

def diff_shadertoy(w : In[int], h : In[int], 
                    cur_col1 : In[Vec3], cur_col2 : In[Vec3],
                    target_col1 : In[Vec3], target_col2 : In[Vec3],
                    loss:Out[float])->Vec6:
   
    y : float = 0.0
    x : float
    gradient_x1: Vec3
    gradient_y1: Vec3
    gradient_z1: Vec3
    gradient_x2: Vec3
    gradient_y2: Vec3
    gradient_z2: Vec3
    #color_cur : Vec3
    color_target : Vec3
    d_color : Diff[Vec3]
    xw_ratio : float
    d_xw_ratio : _dfloat
    temp : Vec3

    d_cur_col1 : Diff[Vec3]
    d_cur_col2 : Diff[Vec3] 
    d_target_col1 : Diff[Vec3] 
    d_target_col2 : Diff[Vec3] 
    
    while (y < h, max_iter := 4096):
        x = 0
        while (x < w, max_iter := 4096):
            xw_ratio = x / w
            d_xw_ratio = make__dfloat(xw_ratio, 0)
            color_target = ray_color(xw_ratio, target_col1, target_col2)

            #respect to col1.x
            d_cur_col1.x = make__dfloat(cur_col1.x, 1)
            d_cur_col1.y = make__dfloat(cur_col1.y, 0)
            d_cur_col1.z = make__dfloat(cur_col1.z, 0)

            d_cur_col2.x = make__dfloat(cur_col2.x, 0)
            d_cur_col2.y = make__dfloat(cur_col2.y, 0)
            d_cur_col2.z = make__dfloat(cur_col2.z, 0)

            d_target_col1.x = make__dfloat(target_col1.x, 1)
            d_target_col1.y = make__dfloat(target_col1.y, 0)
            d_target_col1.z = make__dfloat(target_col1.z, 0)

            d_target_col2.x = make__dfloat(target_col2.x, 0)
            d_target_col2.y = make__dfloat(target_col2.y, 0)
            d_target_col2.z = make__dfloat(target_col2.z, 0)
            
            d_color = d_ray_color(d_xw_ratio, d_cur_col1, d_cur_col2)

            gradient_x1.x = gradient_x1.x+d_color.x.dval*2*(d_color.x.val-color_target.x)            
            gradient_x1.y = gradient_x1.y+d_color.y.dval*2*(d_color.y.val-color_target.y)     
            gradient_x1.z = gradient_x1.z+d_color.z.dval*2*(d_color.z.val-color_target.z)
            
            #respect to col1.y
            d_cur_col1.x = make__dfloat(cur_col1.x, 0)
            d_cur_col1.y = make__dfloat(cur_col1.y, 1)
            d_cur_col1.z = make__dfloat(cur_col1.z, 0)

            d_target_col1.x = make__dfloat(target_col1.x, 0)
            d_target_col1.y = make__dfloat(target_col1.y, 1)
            d_target_col1.z = make__dfloat(target_col1.z, 0)

            d_color = d_ray_color(d_xw_ratio, d_cur_col1, d_cur_col2)

            gradient_y1.x = gradient_y1.x+d_color.x.dval*2*(d_color.x.val-color_target.x)            
            gradient_y1.y = gradient_y1.y+d_color.y.dval*2*(d_color.y.val-color_target.y)     
            gradient_y1.z = gradient_y1.z+d_color.z.dval*2*(d_color.z.val-color_target.z)

            #respect to col1.z
            d_cur_col1.x = make__dfloat(cur_col1.x, 0)
            d_cur_col1.y = make__dfloat(cur_col1.y, 0)
            d_cur_col1.z = make__dfloat(cur_col1.z, 1)

            d_target_col1.x = make__dfloat(target_col1.x, 0)
            d_target_col1.y = make__dfloat(target_col1.y, 0)
            d_target_col1.z = make__dfloat(target_col1.z, 1)

            d_color = d_ray_color(d_xw_ratio, d_cur_col1, d_cur_col2)

            gradient_z1.x = gradient_z1.x+d_color.x.dval*2*(d_color.x.val-color_target.x)            
            gradient_z1.y = gradient_z1.y+d_color.y.dval*2*(d_color.y.val-color_target.y)     
            gradient_z1.z = gradient_z1.z+d_color.z.dval*2*(d_color.z.val-color_target.z)

            #respect to col2.x
            d_cur_col1.x = make__dfloat(cur_col1.x, 0)
            d_cur_col1.y = make__dfloat(cur_col1.y, 0)
            d_cur_col1.z = make__dfloat(cur_col1.z, 0)

            d_cur_col2.x = make__dfloat(cur_col2.x, 1)
            d_cur_col2.y = make__dfloat(cur_col2.y, 0)
            d_cur_col2.z = make__dfloat(cur_col2.z, 0)

            d_target_col1.x = make__dfloat(target_col1.x, 0)
            d_target_col1.y = make__dfloat(target_col1.y, 0)
            d_target_col1.z = make__dfloat(target_col1.z, 0)

            d_target_col2.x = make__dfloat(target_col2.x, 1)
            d_target_col2.y = make__dfloat(target_col2.y, 0)
            d_target_col2.z = make__dfloat(target_col2.z, 0)

            d_color = d_ray_color(d_xw_ratio, d_cur_col1, d_cur_col2)

            gradient_x2.x = gradient_x2.x+d_color.x.dval*2*(d_color.x.val-color_target.x)            
            gradient_x2.y = gradient_x2.y+d_color.y.dval*2*(d_color.y.val-color_target.y)     
            gradient_x2.z = gradient_x2.z+d_color.z.dval*2*(d_color.z.val-color_target.z)
            
            #respect to col2.y
            d_cur_col2.x = make__dfloat(cur_col2.x, 0)
            d_cur_col2.y = make__dfloat(cur_col2.y, 1)
            d_cur_col2.z = make__dfloat(cur_col2.z, 0)

            d_target_col2.x = make__dfloat(target_col2.x, 0)
            d_target_col2.y = make__dfloat(target_col2.y, 1)
            d_target_col2.z = make__dfloat(target_col2.z, 0)

            d_color = d_ray_color(d_xw_ratio, d_cur_col1, d_cur_col2)

            gradient_y2.x = gradient_y2.x+d_color.x.dval*2*(d_color.x.val-color_target.x)            
            gradient_y2.y = gradient_y2.y+d_color.y.dval*2*(d_color.y.val-color_target.y)     
            gradient_y2.z = gradient_y2.z+d_color.z.dval*2*(d_color.z.val-color_target.z)

            #respect to col2.z
            d_cur_col2.x = make__dfloat(cur_col2.x, 0)
            d_cur_col2.y = make__dfloat(cur_col2.y, 0)
            d_cur_col2.z = make__dfloat(cur_col2.z, 1)

            d_target_col2.x = make__dfloat(target_col2.x, 0)
            d_target_col2.y = make__dfloat(target_col2.y, 0)
            d_target_col2.z = make__dfloat(target_col2.z, 1)

            d_color = d_ray_color(d_xw_ratio, d_cur_col1, d_cur_col2)

            gradient_z2.x = gradient_z2.x+d_color.x.dval*2*(d_color.x.val-color_target.x)            
            gradient_z2.y = gradient_z2.y+d_color.y.dval*2*(d_color.y.val-color_target.y)     
            gradient_z2.z = gradient_z2.z+d_color.z.dval*2*(d_color.z.val-color_target.z)

            #compute loss
            loss = loss+ (d_color.x.val-color_target.x)*(d_color.x.val-color_target.x)
            loss = loss+ (d_color.y.val-color_target.y)*(d_color.y.val-color_target.y)
            loss = loss+ (d_color.z.val-color_target.z)*(d_color.z.val-color_target.z)

            x = x + 1
        y = y + 1
    
    gradient_x1.x = gradient_x1.x/(w*h)
    gradient_x1.y = gradient_x1.y/(w*h)
    gradient_x1.z = gradient_x1.z/(w*h)

    gradient_y1.x = gradient_y1.x/(w*h)
    gradient_y1.y = gradient_y1.y/(w*h)
    gradient_y1.z = gradient_y1.z/(w*h)

    gradient_z1.x = gradient_z1.x/(w*h)
    gradient_z1.y = gradient_z1.y/(w*h)
    gradient_z1.z = gradient_z1.z/(w*h)

    gradient_x2.x = gradient_x2.x/(w*h)
    gradient_x2.y = gradient_x2.y/(w*h)
    gradient_x2.z = gradient_x2.z/(w*h)

    gradient_y2.x = gradient_y2.x/(w*h)
    gradient_y2.y = gradient_y2.y/(w*h)
    gradient_y2.z = gradient_y2.z/(w*h)

    gradient_z2.x = gradient_z2.x/(w*h)
    gradient_z2.y = gradient_z2.y/(w*h)
    gradient_z2.z = gradient_z2.z/(w*h)

    return make_vec6(average(gradient_x1), average(gradient_y1), average(gradient_z1),
                     average(gradient_x2), average(gradient_y2), average(gradient_z2))


# def diff_shadertoy(w : In[int], h : In[int], cur_img : In[Array[Vec3]], target_img : In[Array[Vec3]], loss : Out[Array[Vec3]]):
#     y : int = 0
#     x : int
#     d_color : Diff[Vec3]
#     d_col: Diff[Vec3]
#     while (y < h, max_iter := 4096):
#         x = 0
#         while (x < w, max_iter := 4096):
#             d_col.x.val = cur_img[w * y + x].x
#             d_col.y.val = cur_img[w * y + x].y
#             d_col.z.val = cur_img[w * y + x].z
#             d_col.x.dval = 1
#             d_col.y.dval = 1
#             d_col.z.dval = 1
#             d_color = d_shadertoy(d_col)
#             loss[w * y + x].x = loss[w * y + x].x+d_color.x.dval*2*(d_color.x.val-target_img[w * y + x].x)
#             loss[w * y + x].y = loss[w * y + x].y+d_color.y.dval*2*(d_color.y.val-target_img[w * y + x].y)
#             loss[w * y + x].z = loss[w * y + x].z+d_color.z.dval*2*(d_color.z.val-target_img[w * y + x].z)
#             x = x + 1
#         y = y + 1


