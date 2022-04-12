from math import sqrt
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt


LIGHT_AMBIENT = 0.2
F_CONST = 25

def Point(x, y, z):
	return np.array((-y, -x, z))

def distance(p1, p2):
	return norm(p1 - p2)

def normalize(vector):
	return vector / norm(vector)

class Sphere:
	"""Sphere object class"""

	def __init__(self, centre, radius, color):
		self.centre = np.array(centre)
		self.radius = radius
		self.color = np.array(color)

	def __str__(self):
		return f"centre: ({self.centre[0]}, {self.centre[1]}, {self.centre[2]}) r = {self.radius}"

	def normal(self, point):
		return normalize(point - self.centre)

	def intersection(self, ray):
		"""Return point of intersection of ray and sphere or None if they dont intersect"""
		v = ray.r0 - self.centre
		p = np.dot(ray.dir, v)
		q = np.dot(v, v) - self.radius**2
		delta = p**2 - q
		if delta > 0:
			temp = sqrt(delta)
			t1 = -p + temp
			t2 = -p - temp
			if t1 > 0 and t2 > 0:
				ray.hit = ray.r0 + min(t1, t2) * ray.dir


class Ray:
	"""Ray class for ray tracing"""

	def __init__(self, source, direction):
		"""Initialise ray with p0 as sourse and direction, hit postion as (0, 0, 0)"""
		RSTEP = 10**-5
		self.dir = np.array(direction) / np.linalg.norm(direction)
		self.r0 = np.array(source) + RSTEP * self.dir
		self.hit = None
		self.index = None
		self.dist = 0

	def __str__(self):
		return f"{self.r0} + t{self.dir}"

	def distance_from_hit_t(self, hit_t):
		"""Return distance from r0 and ray at hit_t"""
		return np.linalg.norm(self.r0 - (self.r0 + hit_t*self.dir))


	def distance_from_hit_point(self):
		if self.hit is not None:
			return np.linalg.norm(self.hit - self.r0)


	def closest_intersection(self, objects):
		"""Return index of object closest intersect point"""
		closest_obj, closest_dist = None, float("inf")
		for i, obj in enumerate(objects):
			obj.intersection(self)
			if self.hit is not None:
				int_dist = self.distance_from_hit_point()
				if closest_dist > int_dist:
					closest_obj, closest_dist = i, int_dist
		self.index = closest_obj
		self.dist = closest_dist


class RayTracedImage:
	"""2D grid of pixels used to genourate image"""

	def __init__(self, width, height, eyedist, background, lightpos, objects):
		"""Initialize with a x in (-1, 1) and y in (-1/aspect, 1/aspect) in the z-y plane."""
		self.width = width
		self.height = height
		ratio = float(width) / height
		xmin, xmax = -1, 1
		ymin, ymax = -1 / ratio, 1 / ratio
		self.cellx = (xmax - xmin) / width
		self.celly = (ymax - ymin) / height
		self.xmin = xmin
		self.ymin = ymin
		self.eyedist = eyedist
		self.objects = objects
		self.background = np.array(background)
		self.img_grid = np.zeros((width, height, 3))
		self.lightpos = np.array(lightpos)
		

	def set_pixel_color(self, i, j, color):
		"""set the pixel at img[i][j] to color"""
		self.img_grid[i][j][0] = color[0]
		self.img_grid[i][j][1] = color[1]
		self.img_grid[i][j][2] = color[2]


	def lighting(self, ray, obj):
		normal = obj.normal(ray.hit)
		to_light = normalize(self.lightpos - ray.hit)
		h = normalize(to_light + -ray.dir)
		ambient_term = LIGHT_AMBIENT * obj.color
		diffuse_term = np.dot(to_light, normal) * obj.color
		specular_term = np.dot(h, normal)**F_CONST * obj.color
		return ambient_term + diffuse_term + specular_term


	def is_shadow(self, hit, obj):
		to_light = normalize(self.lightpos - obj.normal(hit))
		shadow_ray = Ray(hit, to_light)
		shadow_ray.closest_intersection(self.objects)
		if shadow_ray.hit is not None:
			if distance(self.lightpos, hit) > distance(hit, shadow_ray.hit):
				return True
			else:
				return False
		else:
			return False


	def trace(self, ray):
		"""Compute color value of pixel that ray passes through"""
		color_out = None
		ray.closest_intersection(self.objects)
		if ray.hit is not None:
			obj = self.objects[ray.index]
			if self.is_shadow(ray.hit, obj):
				color_out = LIGHT_AMBIENT * obj.color
			else:
				color_out = self.lighting(ray, obj)
		return color_out
		

	def genourate(self):
		"""Create image data given the objects in view"""
		casts = 0
		zcentre = -self.eyedist
		for i in range(self.width):
			xcentre = self.xmin + (i + 0.5) * self.cellx
			for j in range(self.height):
				ycentre = self.ymin + (j + 0.5) * self.celly
				ray = Ray((0, 0, 0), (xcentre, ycentre, zcentre))
				color = self.trace(ray)
				if color is not None:
					self.img_grid[i][j] = np.clip(color, 0, 1)
				else:
					self.img_grid[i][j] = self.background
				casts += 1
			print(f"Progress: {(casts / (self.width*self.height))*100:.2f}%")
		plt.imsave("img.png", self.img_grid)


s1 = Sphere(Point(0, 0, -5), 2, (1, 0.2, 0))
s2 = Sphere(Point(0, 1, -3), 1/2, (0, 0.5, 1)) 
eyedist = 1
background = (0.5, 0.5, 0.5)
lightpos = Point(0, 2, 0)

img = RayTracedImage(50, 100, eyedist, background, lightpos, [s1, s2])
img.genourate()
