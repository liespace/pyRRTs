from typing import Any
import numpy as np
import reeds_shepp
from plan import BaseRRT


class RRTStar(BaseRRT):
    def __init__(self):
        super(RRTStar, self).__init__()
        self.vertices = None
        self.grid_map = None
        self.grid_res = None
        self.obstacle = None
        self.heuristic = None
        self.maximum_curvature = 0.2

    def preset(self, start, goal, **kwargs):  # type: (BaseRRT.StateNode, BaseRRT.StateNode, Any) -> BaseRRT
        """
        :param start: the start state
        :param goal: the goal state
        :param kwargs: {grid_map, grid_res, obstacle, heuristic}.
            grid_map (numpy.array): occupancy grid map,
            grid_res (float): resolution of grid map,
            obstacle (uint): the value represent the obstacles.
            heuristic (List[Tuple[Tuple]]): [(state, biasing, form)], sampling heuristic path.
                state: state (x_o, y_o, a_o) of the point of the path.
                form==0: biasing = (x_mu, x_sigma), (y_mu, y_sigma), (a_mu, a_sigma).
                form==1: biasing = (r_mu, r_sigma), (theta_mu, theta_sigma), (a_mu, a_sigma).
        :return: RRTStar object
        """
        self.grid_map, self.grid_res = kwargs['grid_map'], kwargs['grid_res']
        self.heuristic = kwargs['heuristic']
        self.start, self.goal = start, goal
        self.start.g = self.goal.h = 0
        self.start.h = self.goal.g = self.cost(start, goal) if self.collision_free(start, goal) else 0
        self.root, self.vertices = self.start, [self.start]
        return self

    def sample_free(self, n):  # type: (int) -> BaseRRT.StateNode
        i = n % len(self.heuristic)
        (state, biasing, form) = self.heuristic[i]
        rand = [state[0], state[1], state[2]]  # [x_o, y_o, a_o]
        if form == 0:
            (x_mu, x_sigma), (y_mu, y_sigma), (a_mu, a_sigma) = biasing
            rand[0] = state[0] + np.random.normal(x_mu, x_sigma, 1)
            rand[1] = state[1] + np.random.normal(y_mu, y_sigma, 1)
            rand[2] = state[2] + np.random.normal(a_mu, a_sigma, 1)
        else:
            (r_mu, r_sigma), (t_mu, t_sigma), (a_mu, a_sigma) = biasing
            r, theta = np.random.normal(r_mu, r_sigma), np.random.normal(t_mu, t_sigma) + state[2]
            rand[0] = state[0] + r * np.cos(theta)
            rand[1] = state[1] + r * np.sin(theta)
            rand[2] = state[2] + np.random.normal(a_mu, a_sigma, 1)
        return BaseRRT.StateNode(tuple(rand))

    def nearest(self, x_rand):  # type: (BaseRRT.StateNode) -> BaseRRT.StateNode
        def cost_to_go(x):
            return self.cost(x, x_rand)
        costs = list(map(cost_to_go, self.vertices))
        return self.vertices[np.argmax(costs)]

    def collision_free(self, x_from, x_to):  # type: (BaseRRT.StateNode, BaseRRT.StateNode) -> bool
        pass

    def attach(self, x_nearest, x_new):  # type: (BaseRRT.StateNode, BaseRRT.StateNode) -> None
        pass

    def rewire(self, x_new):  # type: (BaseRRT.StateNode) -> None
        pass

    def cost(self, x_from, x_to):  # type: (BaseRRT.StateNode, BaseRRT.StateNode) -> float
        return reeds_shepp.path_length(x_from.state, x_to.state, 1. / self.maximum_curvature)

    def path(self):
        pass
