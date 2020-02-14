#!/usr/bin/env python
from copy import deepcopy
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt


def lcs2gcs(self, circle):
    xo, yo, ao = circle.x, circle.y, circle.a
    x = self.x * np.cos(ao) - self.y * np.sin(ao) + xo
    y = self.x * np.sin(ao) + self.y * np.cos(ao) + yo
    a = self.a + ao
    self.x, self.y, self.a = x, y, a


def transform(pts, pto):
    xyo = np.array([[pto[0]], [pto[1]]])
    rot = np.array([[np.cos(pto[2]), -np.sin(pto[2])], [np.sin(pto[2]), np.cos(pto[2])]])
    return np.dot(rot, pts) + xyo


def gcs2lcs(pt, pto):
    xo, yo, ao = pto[0], pto[1], pto[2]
    x = (pt[0] - xo) * np.cos(ao) + (pt[1] - yo) * np.sin(ao)
    y = -(pt[0] - xo) * np.sin(ao) + (pt[1] - yo) * np.cos(ao)
    a = pt[2] - ao
    return x, y, a


def read_task(filepath, seq=0):
    """
    read source(start) and target(goal), and transform to right-hand and local coordinate system centered in source
    LCS: local coordinate system, or said vehicle-frame.
    GCS: global coordinate system
    """
    # read task and transform coordinate system to right-hand
    task = np.loadtxt('{}/{}_task.txt'.format(filepath, seq), delimiter=',')
    org, aim = task[0], task[1]
    source = (org[0], -org[1], -np.radians(org[3]))  # coordinate of start in GCS
    target = (aim[0], -aim[1], -np.radians(aim[3]))  # coordinate of goal in GCS
    # transform source and target coordinate from GCS to LCS.
    start = (0, 0, 0)  # coordinate of start in LCS
    goal = gcs2lcs(target, source)  # coordinate of goal in LCS
    return (source, target), (start, goal)


def read_grid(filepath, seq):
    # type: (str, int) -> np.ndarray
    """read occupancy grid map"""
    return cv2.imread('{}/{}_gridmap.png'.format(filepath, seq), flags=-1)


def main():
    filepath, seq = './test_scenes', 0
    (source, target), (start, goal) = read_task(filepath, seq)
    grid_map = read_grid(filepath, seq)

    contour = np.array([[-2.5, 1.1-1.0], [-(2.5-0.4), 1.1], [2.5-0.6, 1.1], [2.5, 1.1-0.8],
                        [2.5, -(1.1-0.8)], [2.5-0.6, -1.1], [-(2.5-0.4), -1.1], [-2.5, -(1.1-1.0)]])
    contour1 = transform(contour.transpose(), goal).transpose()
    contour1 = np.floor(contour1 / 0.1 + 600 / 2.).astype(int)
    contour2 = transform(contour.transpose(), np.array(goal) + np.array([-2.0, 2.0, 0])).transpose()
    contour2 = np.floor(contour2 / 0.1 + 600 / 2.).astype(int)
    contour3 = transform(contour.transpose(), np.array(start) + np.array([10.0, -10.0, 0])).transpose()
    contour3 = np.floor(contour3 / 0.1 + 600 / 2.).astype(int)

    mask = np.zeros_like(grid_map, dtype=np.uint8)
    # cv2.polylines(mask, np.array([contour]), 1, 255)
    past = time.time()
    cv2.fillPoly(mask, [contour1, contour2, contour3], 255)
    mode = np.zeros((600 + 2, 600 + 2), np.uint8)
    miss = mask.copy()
    cv2.floodFill(miss, mode, (0, 0), 255)
    now = time.time()
    print ((now - past) * 1000)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((50, 50), np.uint8))
    # mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)
    mask = mask | np.bitwise_not(miss)
    result = np.bitwise_and(mask, grid_map)
    print (np.any(result==127))
    cv2.imshow("Mask", result)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
