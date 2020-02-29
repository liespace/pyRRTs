#!/usr/bin/env python
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Polygon
from rrts.planner import RRTStar, BiRRTStar
from rrts.debugger import Debugger


def center2rear(node, wheelbase=2.96):
    """calculate the coordinate of rear track center according to mass center"""
    if not isinstance(node, RRTStar.StateNode):
        theta, r = node[2] + np.pi, wheelbase / 2.
        node[0] += r * np.cos(theta)
        node[1] += r * np.sin(theta)
        return node
    theta, r = node.state[2] + np.pi, wheelbase / 2.
    node.state[0] += r * np.cos(theta)
    node.state[1] += r * np.sin(theta)
    return node


def contour(wheelbase=2.96, width=2.0, length=5.0):  # 2.96, 2.2, 5.0
    return np.array([
        [-(length/2. - wheelbase / 2.), width/2. - 1.0], [-(length/2. - wheelbase / 2. - 0.4), width/2.],
        [length/2. + wheelbase / 2. - 0.6, width/2.], [length/2. + wheelbase / 2., width/2. - 0.8],
        [length/2. + wheelbase / 2., -(width/2. - 0.8)], [length/2. + wheelbase / 2. - 0.6, -width/2.],
        [-(length/2. - wheelbase / 2. - 0.4), -width/2.], [-(length/2. - wheelbase / 2.), -(width/2. - 1.0)]])


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
    return cv2.imread(filename='{}/{}_gridmap.png'.format(filepath, seq), flags=-1)


def read_ose(filepath, seq):
    """read heuristic ose"""
    oseh = np.loadtxt('{}/{}_ose.txt'.format(filepath, seq), delimiter=',')
    oseh = [((x[0], x[1], x[2]), ((0., x[3]/3.), (0., x[3]/3.), (0., x[3]/3. * np.pi/3./3.))) for x in oseh]
    return oseh


def read_yips(filepath, seq, discrimination=0.7):
    yips = np.loadtxt('{}/{}_pred.txt'.format(filepath, seq), delimiter=',')
    yips = filter(lambda x: x[-1] > discrimination, yips)
    yips = map(center2rear, yips)
    yips = [((yip[0], yip[1], yip[2]), ((0.621, 2.146), (0.015, 1.951 * 1.0), (0.005, 0.401 * 1.0))) for yip in yips]
    return yips


def set_plot(switch=True):
    if switch:
        plt.ion()
        plt.figure()
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.gca().set_aspect('equal')
        plt.gca().set_facecolor((0.2, 0.2, 0.2))
        plt.gca().set_xlim((-30, 30))
        plt.gca().set_ylim((-30, 30))
        plt.draw()


def transform(pts, pto):
    xyo = np.array([[pto[0]], [pto[1]]])
    rot = np.array([[np.cos(pto[2]), -np.sin(pto[2])], [np.sin(pto[2]), np.cos(pto[2])]])
    return np.dot(rot, pts) + xyo


def main():
    filepath, seq, debug = './test_scenes', 2062, True
    rrt_star = BiRRTStar().set_vehicle(contour(), 0.3, 0.25)
    # heuristic = read_ose(filepath, seq)
    # heuristic = read_yips(filepath, seq)
    heuristic = None
    source, target = read_task(filepath, seq)
    start = center2rear(deepcopy(source)).gcs2lcs(source.state)
    goal = center2rear(deepcopy(target)).gcs2lcs(source.state)
    grid_ori = deepcopy(source).gcs2lcs(source.state)
    grid_map = read_grid(filepath, seq)
    grid_res = 0.1

    if debug:
        set_plot(debug)
        Debugger.plot_grid(grid_map, grid_res)
        Debugger().plot_nodes([start, goal])
        plt.gca().add_patch(Polygon(
            transform(contour().transpose(), start.state).transpose(), True, color='b', fill=False, lw=2.0))
        plt.gca().add_patch(Polygon(
            transform(contour().transpose(), goal.state).transpose(), True, color='g', fill=False, lw=2.0))
        if heuristic:
            Debugger.plot_heuristic(heuristic)
        plt.draw()
    rrt_star.debug = debug
    rrt_star.preset(start, goal, grid_map, grid_res, grid_ori, 255, heuristic).planning(100, debug=debug)

    Debugger.breaker('Plotting', switch=debug)


if __name__ == '__main__':
    main()
