from abc import abstractmethod, abstractproperty
from typing import List, Any
import numpy as np


class BaseRRT(object):
    def __init__(self):
        self.root = None
        self.vertices = []
        self.start = None
        self.goal = None

    @abstractmethod
    def preset(self, start, goal, **kwargs):
        # type: (BaseRRT.StateNode, BaseRRT.StateNode, Any) -> BaseRRT
        """
        initialize the parameters for planning: Start State, Goal State and other needs.
        """
        pass

    def planning(self, times):
        # type: (int) -> None
        """
        main flow.
        """
        for i in range(times):
            x_new = self.sample_free(i)
            x_nearest = self.nearest(x_new)
            if self.collision_free(x_nearest, x_new):
                self.attach(x_nearest, x_new)
                self.rewire(x_new)

    @abstractmethod
    def sample_free(self, n):
        # type: (int) -> BaseRRT.StateNode
        """
        sample a state from free configuration space.
        """
        pass

    @abstractmethod
    def nearest(self, x_rand):
        # type: (BaseRRT.StateNode) -> BaseRRT.StateNode
        """
        find the state in the tree which is nearest to the sampled state.
        """
        pass

    @abstractmethod
    def collision_free(self, x_from, x_to):
        # type: (BaseRRT.StateNode, BaseRRT.StateNode) -> bool
        """
        check if the path from one state to another state collides with any obstacles or not.
        """
        pass

    @abstractmethod
    def attach(self, x_nearest, x_new):
        # type: (BaseRRT.StateNode, BaseRRT.StateNode) -> None
        """
        add the new state to the tree and complement other values.
        """
        pass

    @abstractmethod
    def rewire(self, x_new):
        # type: (BaseRRT.StateNode) -> None
        """
        rewiring tree by the new state.
        """
        pass

    @abstractproperty
    def path(self):
        """
        extract the planning result.
        """
        pass

    @property
    def trajectory(self):
        """
        planning velocity for a path to generate a trajectory.
        """
        return None

    class StateNode(object):
        def __init__(self, state=(), g=np.inf, h=np.inf, parent=None, children=None):
            # type: (tuple, BaseRRT.StateNode, float, float, List[BaseRRT.StateNode]) -> None
            """
            :param state: state of the Node, a tuple (x, y, orientation).
            :param g: cost from root to here.
            :param h: cost from here to goal.
            :param parent: its parent node.
            :param children: a list of its children node.
            """
            self.state = np.array(state)
            self.v, self.k = None, None  # velocity of the state, curvature of the state (related to the steering angle)
            self.g, self.h = g, h
            self.f = self.g + self.h
            self.parent = parent
            self.children = children if children else []
            self.status = 0  # 0 for safe, 1 for dangerous.

        def set_parent(self, x_parent):
            # type: (BaseRRT.StateNode) -> None
            """
            add a state as parent.
            """
            self.parent = x_parent
