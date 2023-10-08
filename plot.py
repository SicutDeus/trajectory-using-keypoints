import math
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import matplotlib

plt.ion()
class DynamicUpdate():
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.lim = 1
        self.ax.set_xlim([-self.lim, self.lim])
        self.ax.set_ylim([-self.lim, self.lim])
        self.dx = 0
        self.dy = 0
        self.points = []
        self.lines = []

    def update_lims(self, x, y):
        tmp = self.lim / 1.15
        self.lim = max(math.fabs(tmp), math.fabs(x), math.fabs(y)) * 1.15
        self.ax.set_xlim([-self.lim, self.lim])
        self.ax.set_ylim([-self.lim, self.lim])
    def add_point(self, x, y, z):
        self.points.append((x, y, z))

    def add_line(self, x1, y1, z1, x2, y2, z2):
        self.lines.append(((x1, y1, z1), (x2, y2, z2)))

    def plot(self):
        if len(self.lines) > 0:
            x1, y1, z1 = self.lines[-1][0]
            x2, y2, z2 = self.lines[-1][1]
            self.ax.plot([x1, x2], [y1, y2], [z1, z2])

    def __call__(self, x, y, z, sleep=False, save=False):
        self.plot()
        if save:
            plt.savefig('result.png')
        if sleep:
            plt.savefig('test.png')
            plt.pause(1000)

        elif len(self.points) == 0 or (x, y, z) != self.points[-1]:
            self.add_point(x, y, z)
            self.update_lims(x, y)
            if len(self.points) > 1:
                x1, y1, z1 = self.points[-2]
                x2, y2, z2 = self.points[-1]
                self.points = self.points[1:]
                self.add_line(x1, y1, z1, x2, y2, z2)
            self.plot()
            #plt.pause(0.000001)
        return x, y