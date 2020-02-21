from typing import List, Tuple, Optional, Any
from copy import deepcopy
import numpy as np
import cv2
import reeds_shepp
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Wedge, Polygon


class Debugger(object):

    @staticmethod
    def plot_polygon(ploy, color='b', lw=2., fill=False):
        actor = plt.gca().add_patch(Polygon(ploy, True, color=color, fill=fill, linewidth=lw))
        plt.draw()
        return [actor]

    @staticmethod
    def plot_curve(x_from, x_to, rho, color='g'):
        states = reeds_shepp.path_sample(x_from.state, x_to.state, rho, 0.3)
        x, y = [state[0] for state in states], [state[1] for state in states]
        actor = plt.plot(x, y, c=color)
        plt.draw()
        return [actor]

    def plot_nodes(self, nodes, color=None):
        def plotting(x):
            return self.plot_state(x.state, color if color else (0.5, 0.8, 0.5))
        return map(plotting, nodes)

    @staticmethod
    def plot_state(state, color=(0.5, 0.8, 0.5)):
        cir = plt.Circle(xy=(state[0], state[1]), radius=0.2, color=color, alpha=0.6)
        arr = plt.arrow(x=state[0], y=state[1], dx=0.5 * np.cos(state[2]), dy=0.5 * np.sin(state[2]), width=0.1)
        actors = [plt.gca().add_patch(cir), plt.gca().add_patch(arr)]
        plt.draw()
        return actors

    @staticmethod
    def plot_heuristic(heuristic, color=(0.5, 0.8, 0.5)):
        actors = []
        for item in heuristic:
            state, biasing = item
            cir = Ellipse(xy=(state[0], state[1]), width=biasing[0][1]*2*2,
                          height=biasing[1][1]*2*2, color=color, alpha=0.6, fill=False)
            arr = Wedge(center=(state[0], state[1]), r=1.0, theta1=np.degrees(state[2] - biasing[2][1]*2),
                        theta2=np.degrees(state[2] + biasing[2][1]*2), fill=False, color=color)
            actors.append([plt.gca().add_patch(cir), plt.gca().add_patch(arr)])
        plt.draw()
        return actors

    @staticmethod
    def plot_grid(grid_map, grid_res):
        # type: (np.ndarray, float) -> None
        """plot grid map"""
        row, col = grid_map.shape[0], grid_map.shape[1]
        indexes = np.argwhere(grid_map == 255)
        xy2uv = np.array([[0., 1. / grid_res, row / 2.], [1. / grid_res, 0., col / 2.], [0., 0., 1.]])
        for index in indexes:
            uv = np.array([index[0], index[1], 1])
            xy = np.dot(np.linalg.inv(xy2uv), uv)
            rect = plt.Rectangle((xy[0] - grid_res, xy[1] - grid_res), grid_res, grid_res, color=(1.0, 0.1, 0.1))
            plt.gca().add_patch(rect)
        plt.draw()

    @staticmethod
    def remove(actor):
        for x in actor:
            if isinstance(x, list):
                a, b = x[0], x[1]
                a.remove(), b.remove()
            else:
                x.remove()
        plt.draw()
