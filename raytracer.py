
import numpy as np
import matplotlib.pyplot as plt

class Sphere:
	"""Sphere object class"""

	def __init__(self, centre, radius, color):
		self.centre = np.array(centre)
		self.radius = radius
		self.color = np.array(color)

	def __str__(self):
		return f"centre: ({self.centre[0]}, {self.centre[1]}, {self.centre[2]}) r = {self.radius}"

	def intersection(self, ray):
		"""Return point of intersection of ray and sphere or None if they dont intersect"""
		p = ray.dir[0]**2 + ray.dir[1]**2 + ray.dir[2]**2
		q = 2*ray.dir[0]*(ray.r0[0] - self.centre[0]) + 2*ray.dir[1]*(ray.r0[1] - self.centre[1]) + 2*ray.dir[2]*(ray.r0[2] - self.centre[2])
		w = (ray.r0[0] - self.centre[0])**2 + (ray.r0[1] - self.centre[1])**2 + (ray.r0[2] - self.centre[2])**2 - self.radius**2
		if q**2 - 4 * p * w >= 0:
			hit_t_1 = (-q + np.sqrt(q**2-4*p*w)) / 2*w
			hit_t_2 = (-q - np.sqrt(q**2-4*p*w)) / 2*w
			if ray.distance_from_hit_t(hit_t_1) < ray.distance_from_hit_t(hit_t_2):
				ray.hit = ray.r0 + hit_t_1 * ray.dir
			else:
				ray.hit = ray.r0 + hit_t_2 * ray.dir
		

class Ray:
	"""Ray class for ray tracing"""

	def __init__(self, source, direction):
		"""Initialise ray with p0 as sourse and direction, hit postion as (0, 0, 0)"""
		RSTEP = 10**-7
		self.dir = np.array(direction) / np.linalg.norm(direction)
		self.r0 = np.array(source) + RSTEP * self.dir
		self.hit = None
		self.index = -1
		self.dist = 0


	def distance_from_hit_t(self, hit_t):
		"""Return distance from r0 and ray at hit_t"""
		return np.linalg.norm(self.r0 - (self.r0 + hit_t*self.dir))


	def distance_from_hit_point(self):
		if self.hit is not None:
			return np.linalg.norm(self.hit - self.r0)


	def trace(self, objects):
		"""Return index of object closest intersect point"""
		closest_index, dist = None, None
		for i, obj in enumerate(objects):
			obj.intersection(self)
			if self.hit is not None:
				int_dist = self.distance_from_hit_point()
				if closest_index is None:
					closest_index, dist = i, int_dist
				elif dist > int_dist:
					closest_index, dist = i, int_dist
		if closest_index is not None:
			return objects[closest_index].color
		else:
			return None


class RayTracedImage:
	"""2D grid of pixels used to genourate image"""

	def __init__(self, width, height, background_color):
		"""Initialize with a x in (-1, 1) and y in (-1/aspect, 1/aspect) in the z-y plane."""
		ratio = float(width) / height
		self.xmin, self.xmax = -1, 1
		self.ymin, self.ymax = -1 / ratio, 1/ ratio
		self.xspace = np.linspace(self.xmin, self.xmax, width)
		self.yspace = np.linspace(self.ymin, self.ymax, height)
		self.cellx = (self.xmax - self.xmin) / width
		self.celly = (self.ymax - self.ymin) / height
		self.eyepos = np.array((0, 0, 1))
		self.eyedist = 1
		self.background = np.array(background_color)
		self.img_grid = np.zeros((width, height, 3))


	def set_pixel_color(self, i, j, color):
		"""set the pixel at img[i][j] to color"""
		self.img_grid[i][j][0] = color[0]
		self.img_grid[i][j][1] = color[1]
		self.img_grid[i][j][2] = color[2]


	def genourate(self, objects):
		"""Create image data given the objects in view"""
		for i, x in enumerate(self.xspace):
			xcentre = self.xmin + (i + 0.5) * self.cellx
			for j, y in enumerate(self.yspace):
				ycentre = self.ymin + (j + 0.5) * self.celly
				ray = Ray(self.eyepos, np.array((xcentre, ycentre, 0)) - self.eyepos)
				color = ray.trace(objects)
				if color is None:
					color = self.background
				self.img_grid[i][j] = color
		plt.imsave("img.png", self.img_grid)


img = RayTracedImage(500, 500, (0.5, 0.5, 0.5))
s1 = Sphere((0, 0, -5), 1, (1, 0, 0))
s2 = Sphere((0.5, 1, -2), 0.5, (0, 1, 0))
img.genourate([s1, s2])
