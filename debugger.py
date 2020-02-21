import numpy as np
import reeds_shepp
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Wedge, Polygon


class Debugger(object):
    path_history = []

    def debug_nearest_searching(self, state, switch=True):
        if switch:
            actor = self.plot_state(state, color='r')
            raw_input('nearest node')
            self.remove(actor)

    def debug_planned_path(self, path, no, switch=True):
        if switch:
            self.path_history.append((no, path[-1].fu if path[-1].fu < np.inf else 0))
            np.savetxt('path_history.csv', self.path_history, delimiter=',')
            actor = self.plot_nodes(path, 'r')
            raw_input('Planned Path {}'.format(path[-1].fu))
            self.remove(actor)

    def debug_sample_emerging(self, x_rand, poly, switch=True):
        if switch:
            actor_state = Debugger.plot_state(x_rand)
            actor_poly = Debugger.plot_polygon(self.transform(poly, x_rand))
            raw_input('sample emerged')
            self.remove(actor_poly)
            self.remove(actor_state)

    def debug_sampling(self, state, switch=True):
        if switch:
            actor = self.plot_state(state)
            raw_input('new node')
            self.remove(actor)

    def debug_collision_checking(self, states, poly, result, switch=True):
        if switch:
            actors = [self.plot_polygon(self.transform(poly, state))[0] for state in states]
            words = 'free' if result else 'collided'
            raw_input('collision checked ({})'.format(words))
            self.remove(actors)

    def debug_attaching(self, x_nearest, x_new, rho, switch=True):
        if switch:
            self.plot_curve(x_nearest, x_new, rho)
            raw_input('added new node ({}, {}, {})'.format(x_new.g, x_new.hu, x_new.fu))

    def debug_rewiring_check(self, xs, x_new, switch=True):
        if switch:
            actors = self.plot_nodes(xs, color='r')
            raw_input('rewire checked (g={}, n={})'.format(x_new.g, len(xs)))
            self.remove(actors)

    def debug_rewiring(self, x, cost, switch=True):
        if switch:
            actor_state = self.plot_state(x.state, color='r')
            raw_input('need rewiring {} -> {}'.format(x.g, cost))
            self.remove(actor_state)

    @staticmethod
    def transform(poly, pto):
        pts = poly.transpose()
        xyo = np.array([[pto[0]], [pto[1]]])
        rot = np.array([[np.cos(pto[2]), -np.sin(pto[2])], [np.sin(pto[2]), np.cos(pto[2])]])
        return (np.dot(rot, pts) + xyo).transpose()

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