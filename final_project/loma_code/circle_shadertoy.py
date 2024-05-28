class Vec3:
    x : float
    y : float
    z : float


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

def sphere_isect(sph : In[Sphere], ray : In[Ray]) -> float:
    oc : Vec3 = sub(ray.org, sph.center)
    a : float = dot(ray.dir, ray.dir)
    b : float = 2 * dot(oc, ray.dir)
    c : float = dot(oc, oc) - sph.radius * sph.radius
    discriminant : float = b * b - 4 * a * c
    ret_dist : float = 0
    if discriminant < 0:
        ret_dist = -1
    else:
        ret_dist = (-b - sqrt(discriminant)) / (2 * a)
    return ret_dist

def ray_color(ray : In[Ray], center: In[Vec3], radius: In[float]) -> Vec3:
    sph : Sphere
    sph.center = center
    sph.radius = radius

    ret_color : Vec3
    t : float = sphere_isect(sph, ray)

    N : Vec3
    white : Vec3 = make_vec3(1, 1, 1)
    blue : Vec3 = make_vec3(0.5, 0.7, 1)
    a : float

    if t > 0:
        N = normalize(sub(add(ray.org, mul(t, ray.dir)), sph.center))
        ret_color = make_vec3(0.5 * (N.x + 1), 0.5 * (N.y + 1), 0.5 * (N.z + 1))
    else:
        a = 0.5 * ray.dir.y + 1
        ret_color = add(mul((1 - a), white), mul(a, blue))
    return ret_color

d_ray_color = fwd_diff(ray_color)

def diff_shadertoy(w : In[int], h : In[int], 
                   cur_radius : In[float], target_radius : In[float], 
                   cur_center : In[Vec3], target_center: In[Vec3],
                   image_x : Out[Array[Vec3]],image_y: Out[Array[Vec3]], loss:Out[float])->Vec3:
    # Camera setup
    aspect_ratio : float = int2float(w) / int2float(h)
    focal_length : float = 1.0
    viewport_height : float = 2.0
    viewport_width : float = viewport_height * aspect_ratio
    camera_center : Vec3 = make_vec3(0, 0, 0)
    # Calculate the horizontal and vertical delta vectors from pixel to pixel.
    pixel_delta_u : Vec3 = make_vec3(viewport_width / w, 0, 0)
    pixel_delta_v : Vec3 = make_vec3(0, -viewport_height / h, 0)
    # Calculate the location of the upper left pixel.
    viewport_upper_left : Vec3 = make_vec3(\
            camera_center.x - viewport_width / 2,
            camera_center.y + viewport_height / 2,
            camera_center.z - focal_length
        )
    pixel00_loc : Vec3 = viewport_upper_left
    pixel00_loc.x = pixel00_loc.x + pixel_delta_u.x / 2
    pixel00_loc.y = pixel00_loc.y - pixel_delta_v.y / 2

    y : int = 0
    x : int
    pixel_center : Vec3
    #ray : Ray
    ray: Ray
    d_ray : Diff[Ray]
    ray_dir : Vec3
    d_color : Diff[Vec3]
    color_target : Vec3

    d_cur_radius : Diff[float] = make__dfloat(cur_radius, 0)
    d_cur_center : Diff[Vec3] 
    d_cur_center.z = make__dfloat(-1, 0) # fix z value at -1; we consider 2D center only

    gradient_x: Vec3
    gradient_y: Vec3
    while (y < h, max_iter := 4096):
        x = 0
        while (x < w, max_iter := 4096):
            
            pixel_center = add(add(pixel00_loc, mul(x, pixel_delta_u)), mul(y, pixel_delta_v))
            ray_dir = normalize(sub(pixel_center, camera_center))

            ray.org.x = camera_center.x
            ray.org.y = camera_center.y
            ray.org.z = camera_center.z
            ray.dir.x = ray_dir.x
            ray.dir.y = ray_dir.y
            ray.dir.z = ray_dir.z

            d_ray.org.x.val = camera_center.x
            d_ray.org.y.val = camera_center.y
            d_ray.org.z.val = camera_center.z
            d_ray.dir.x.val = ray_dir.x
            d_ray.dir.y.val = ray_dir.y
            d_ray.dir.z.val = ray_dir.z

            #gt
            color_target = ray_color(ray, target_center, target_radius)

            #respect to x
            d_cur_center.x = make__dfloat(cur_center.x, 1)
            d_cur_center.y = make__dfloat(cur_center.y, 0)
            d_color = d_ray_color(d_ray, d_cur_center, d_cur_radius)
            image_x[w * y + x].x = d_color.x.dval
            image_x[w * y + x].y = d_color.y.dval
            image_x[w * y + x].z = d_color.z.dval

            gradient_x.x = gradient_x.x+d_color.x.dval*2*(d_color.x.val-color_target.x)            
            gradient_x.y = gradient_x.y+d_color.y.dval*2*(d_color.y.val-color_target.y)     
            gradient_x.z = gradient_x.z+d_color.z.dval*2*(d_color.z.val-color_target.z)
            

            #respect to y
            d_cur_center.x = make__dfloat(cur_center.x, 0)
            d_cur_center.y = make__dfloat(cur_center.y, 1)
            d_color = d_ray_color(d_ray, d_cur_center, d_cur_radius)
            
            image_y[w * y + x].x = d_color.x.dval
            image_y[w * y + x].y = d_color.y.dval
            image_y[w * y + x].z = d_color.z.dval

            gradient_y.x = gradient_y.x+d_color.x.dval*2*(d_color.x.val-color_target.x)
            gradient_y.y = gradient_y.y+d_color.y.dval*2*(d_color.y.val-color_target.y)
            gradient_y.z = gradient_y.z+d_color.z.dval*2*(d_color.z.val-color_target.z)

            #compute loss
            loss = loss+ (d_color.x.val-color_target.x)*(d_color.x.val-color_target.x)
            loss = loss+ (d_color.y.val-color_target.y)*(d_color.y.val-color_target.y)
            loss = loss+ (d_color.z.val-color_target.z)*(d_color.z.val-color_target.z)

            x = x + 1
        y = y + 1
    gradient_x.x = gradient_x.x/(w*h)
    gradient_x.y = gradient_x.y/(w*h)
    gradient_x.z = gradient_x.z/(w*h)

    gradient_y.x = gradient_y.x/(w*h)
    gradient_y.y = gradient_y.y/(w*h)
    gradient_y.z = gradient_y.z/(w*h) 
    return make_vec3(average(gradient_x),average(gradient_y),1.0)


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


