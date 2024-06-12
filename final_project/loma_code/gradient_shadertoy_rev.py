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
# void d_ray_color(float xw_ratio, float* _dxw_ratio_eBLyqa, 
#                  Vec3 col1, Vec3* _dcol1_B609tD, 
#                  Vec3 col2, Vec3* _dcol2_tQTyVC, Vec3 _dreturn_oewoBi)

d_ray_color = rev_diff(ray_color)

def diff_shadertoy(w : In[int], h : In[int], 
                    cur_col1 : In[Vec3], cur_col2 : In[Vec3],
                    target_col1 : In[Vec3], target_col2 : In[Vec3],
                    loss:Out[float], image : Out[Array[Vec3]])->Vec6:
    #return make_vec6(0,0,0,0,0,0)
    y : float = 0.0
    x : float
    gradient_x1: Vec3
    gradient_y1: Vec3
    gradient_z1: Vec3
    gradient_x2: Vec3
    gradient_y2: Vec3
    gradient_z2: Vec3
    color_current : Vec3
    color_target : Vec3
    d_color : Vec3
    xw_ratio : float
    d_xw_ratio : float

    d_cur_col1 : Vec3
    d_cur_col2 : Vec3
    d_target_col1 : Vec3 
    d_target_col2 : Vec3
    
    while (y < h, max_iter := 4096):
        x = 0
        while (x < w, max_iter := 4096):
            xw_ratio = x / w
            color_current = ray_color(xw_ratio, cur_col1, cur_col2)
            color_target = ray_color(xw_ratio, target_col1, target_col2)

            # plot the actual current color
            image[float2int(w * y + x)].x = color_current.x
            image[float2int(w * y + x)].y = color_current.y
            image[float2int(w * y + x)].z = color_current.z

            
            d_ray_color(xw_ratio, d_xw_ratio, cur_col1, d_cur_col1, cur_col2, d_cur_col2, make_vec3(1.0,1.0,1.0))

            # respect to col1.x
            gradient_x1.x = gradient_x1.x+d_cur_col1.x*2*(color_current.x-color_target.x)            
            gradient_x1.y = gradient_x1.y+d_cur_col1.x*2*(color_current.y-color_target.y)     
            gradient_x1.z = gradient_x1.z+d_cur_col1.x*2*(color_current.z-color_target.z)
            
            # #respect to col1.y
            gradient_y1.x = gradient_y1.x+d_cur_col1.y*2*(color_current.x-color_target.x)            
            gradient_y1.y = gradient_y1.y+d_cur_col1.y*2*(color_current.y-color_target.y)     
            gradient_y1.z = gradient_y1.z+d_cur_col1.y*2*(color_current.z-color_target.z)

            # #respect to col1.z
            gradient_z1.x = gradient_z1.x+d_cur_col1.z*2*(color_current.x-color_target.x)            
            gradient_z1.y = gradient_z1.y+d_cur_col1.z*2*(color_current.y-color_target.y)     
            gradient_z1.z = gradient_z1.z+d_cur_col1.z*2*(color_current.z-color_target.z)

            #respect to col2.x
            gradient_x2.x = gradient_x2.x+d_cur_col2.x*2*(color_current.x-color_target.x)            
            gradient_x2.y = gradient_x2.y+d_cur_col2.x*2*(color_current.y-color_target.y)     
            gradient_x2.z = gradient_x2.z+d_cur_col2.x*2*(color_current.z-color_target.z)
            
            # #respect to col2.y
            gradient_y2.x = gradient_y2.x+d_cur_col2.y*2*(color_current.x-color_target.x)            
            gradient_y2.y = gradient_y2.y+d_cur_col2.y*2*(color_current.y-color_target.y)     
            gradient_y2.z = gradient_y2.z+d_cur_col2.y*2*(color_current.z-color_target.z)

            # #respect to col2.z
            gradient_z2.x = gradient_z2.x+d_cur_col2.z*2*(color_current.x-color_target.x)            
            gradient_z2.y = gradient_z2.y+d_cur_col2.z*2*(color_current.y-color_target.y)     
            gradient_z2.z = gradient_z2.z+d_cur_col2.z*2*(color_current.z-color_target.z)

            #compute loss
            loss = loss+ (color_current.x-color_target.x)*(color_current.x-color_target.x)
            loss = loss+ (color_current.y-color_target.y)*(color_current.y-color_target.y)
            loss = loss+ (color_current.z-color_target.z)*(color_current.z-color_target.z)

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



