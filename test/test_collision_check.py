#!/usr/bin/env python
from copy import deepcopy
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from planner import RRTStar
import reeds_shepp


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


def transform(poly, pto):
    pts = poly.transpose()
    xyo = np.array([[pto[0]], [pto[1]]])
    rot = np.array([[np.cos(pto[2]), -np.sin(pto[2])], [np.sin(pto[2]), np.cos(pto[2])]])
    return (np.dot(rot, pts) + xyo).transpose()


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
    return cv2.imread('{}/{}_gridmap.png'.format(filepath, seq), flags=-1)


def main():
    filepath, seq = './test_scenes', 0
    source, target = read_task(filepath, seq)
    grid_map = read_grid(filepath, seq)
    start = center2rear(deepcopy(source).gcs2lcs(source.state))
    goal = center2rear(deepcopy(target).gcs2lcs(source.state))

    states = reeds_shepp.path_sample(start.state, goal.state, 5.0, 0.3)
    cons = [transform(contour(), state) for state in states]
    cons = [np.floor(con / 0.1 + 600 / 2.).astype(int) for con in cons]

    mask = np.zeros_like(grid_map, dtype=np.uint8)
    past = time.time()
    [cv2.fillPoly(mask, [con], 255) for con in cons]
    # mode = np.zeros((600 + 2, 600 + 2), np.uint8)
    # miss = mask.copy()
    # cv2.floodFill(miss, mode, (0, 0), 255)
    now = time.time()
    print ((now - past) * 1000)

    # mask = mask | np.bitwise_not(miss)
    # result = np.bitwise_and(mask, grid_map)
    # print (np.any(result==127))
    result = np.bitwise_and(mask, grid_map)
    print (grid_map.min())
    print (np.all(result < 255))

    cv2.imshow("Mask", mask)

    cv2.waitKey(0)


if __name__ == '__main__':
    main()
