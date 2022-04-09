import numpy as np
import matplotlib.pyplot as plt
from random import uniform

class Ray:
	"""Ray class for ray tracing"""

	def __init__(self, source, direction):
		"""Initalise ray with p0 as sourse and direction, hit postion as (0, 0, 0)"""
		RSTEP = 10**-7
		self.dir = direction / np.linalg.norm(direction)
		self.p0 = sourse + RSTEP * self.dir
		self.hit = np.zeros(3)
		self.index = -1
		self.dist = 0

	def trace(self, objects):
		"""return RGB color value of traced ray"""
		pass



class RayTracedImage:
	"""2D grid of pixels used to genourate image"""

	def __init__(self, width, height):
		"""Initalize with a x in (-1, 1) and y in (-1/aspect, 1/aspect) in the z-y plane."""
		ratio = float(width) / height
		self.xmin, self.xmax = -1, 1
		self.ymin, self.ymax = -1 / ratio, 1/ ratio
		self.xspace = np.linspace(self.xmin, self.xmax, width)
		self.yspace = np.linspace(self.ymin, self.ymax, height)
		self.cellx = (self.xmax - self.xmin) / width
		self.celly = (self.ymin - self.ymin) / height
		self.eyepos = np.array((0, 0, 1))
		self.eyedist = 1
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
				ycentre = (self.ymax - self.xmin) + (j + 0.5) * self.celly
				self.set_pixel_color(i, j, (uniform(0, 1), uniform(0, 1), uniform(0, 1)))


		plt.imsave("img.png", self.img_grid)



img = RayTracedImage(500, 500)
img.genourate("s")