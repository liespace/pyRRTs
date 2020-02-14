from abc import abstractmethod, abstractproperty
from typing import List, Any
import numpy as np


class BaseRRT(object):
    def __init__(self):
        self.root = None
        self.start = None
        self.goal = None

    @abstractmethod
    def preset(self, start, goal, **kwargs):
        # type: (BaseRRT.StateNode, BaseRRT.StateNode, Any) -> BaseRRT
        """
        initialize the parameters for planning: Start State, Goal State and other needs.
        """
        pass

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
    def cost(self, x_from, x_to):
        # type: (BaseRRT.StateNode, BaseRRT.StateNode) -> float
        """calculate the cost from one state to another state"""
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
