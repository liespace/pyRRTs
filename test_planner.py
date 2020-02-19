#!/usr/bin/env python
from copy import deepcopy
import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from planner import RRTStar


def center2rear(node, wheelbase=2.96):  # type: (RRTStar.StateNode, float) -> RRTStar.StateNode
    """calculate the coordinate of rear track center according to mass center"""
    theta, r = node.state[2] + np.pi, wheelbase / 2.
    node.state[0] += r * np.cos(theta)
    node.state[1] += r * np.sin(theta)
    return node


def contour(wheelbase=2.96):
    return np.array([
        [-(2.5 - wheelbase/2.), 1.1 - 1.0], [-(2.5 - wheelbase/2. - 0.4), 1.1],
        [2.5 + wheelbase/2. - 0.6, 1.1], [2.5 + wheelbase/2., 1.1 - 0.8],
        [2.5 + wheelbase/2., -(1.1 - 0.8)], [2.5 + wheelbase/2. - 0.6, -1.1],
        [-(2.5 - wheelbase/2. - 0.4), -1.1], [-(2.5 - wheelbase/2.), -(1.1 - 1.0)]])


def read_task(filepath, seq=0):
    """
    read source(start) and target(goal), and transform to right-hand and local coordinate system centered in source
    LCS: local coordinate system, or said vehicle-frame.
    GCS: global coordinate system
    """
    # read task and transform coordinate system to right-hand
    task = np.loadtxt('{}/{}_task.txt'.format(filepath, seq), delimiter=',')
    org, aim = task[0], task[1]
    # coordinate of the center of mass on source(start) state, in GCS
    source = RRTStar.StateNode(state=(org[0], -org[1], -np.radians(org[3])))
    # coordinate of center of mass on target(goal) state, in GCS
    target = RRTStar.StateNode(state=(aim[0], -aim[1], -np.radians(aim[3])))
    return source, target


def read_grid(filepath, seq):
    # type: (str, int) -> np.ndarray
    """read occupancy grid map"""
    return np.array(Image.open('{}/{}_gridmap.png'.format(filepath, seq)))


def read_ose(filepath, seq):
    """read heuristic ose"""
    oseh = np.loadtxt('{}/{}_ose.txt'.format(filepath, seq), delimiter=',')
    oseh = [((x[0], x[1], x[2]), ((0., x[3]/3.), (0., 2*np.pi/3.), (0., np.pi/4./3.)), 1) for x in oseh]
    return oseh


def read_yips(filepath, seq, discrimination=0.7):
    yips = np.loadtxt('{}/{}_pred.txt'.format(filepath, seq), delimiter=',')
    yips = filter(lambda x: x[-1] > discrimination, yips)
    yips = [((yip[0], yip[1], yip[2]), ((0.621, 2.146), (0.015, 1.951), (0.005, 0.401)), 0) for yip in yips]
    return yips


def set_plot(rrt_star):
    # type: (RRTStar) -> None
    plt.ion()
    plt.figure()
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.gca().set_aspect('equal')
    plt.gca().set_facecolor((0.2, 0.2, 0.2))
    plt.gca().set_xlim((-30, 30))
    plt.gca().set_ylim((-30, 30))
    rrt_star.plot_grid(rrt_star.grid_map, rrt_star.grid_res)
    rrt_star.plot_nodes([rrt_star.start, rrt_star.goal])
    plt.draw()


def main():
    filepath, seq = './test_scenes', 0
    heuristic = read_ose(filepath, seq)
    # heuristic = read_yips(filepath, seq)
    state, biasing, form = heuristic[0]
    print (len(heuristic))
    source, target = read_task(filepath, seq)
    start = center2rear(deepcopy(source)).gcs2lcs(source.state)
    goal = center2rear(deepcopy(target)).gcs2lcs(source.state)
    grid_ori = deepcopy(source).gcs2lcs(source.state)
    grid_map = read_grid(filepath, seq)
    grid_res = 0.1
    rrt_star = RRTStar().set_vehicle(contour(), 0.3, 0.25)
    rrt_star.preset(start, goal, grid_map, grid_res, grid_ori, 255, heuristic)


if __name__ == '__main__':
    main()
