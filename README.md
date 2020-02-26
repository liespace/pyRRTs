# RRTs
## What is it
A repository of Python2 implemented RRT\*-based algorithms for Path Planning of Autonomous Driving. 

Currently, it includes these variants:

1. RRT\*[^1], for static environments (parking lots, narrow space).
2. Bi-RRT\*[^2], for static environments (parking lots, narrow space).

## How to use
```python
from rrts.planner import RRTStar, BiRRTStar
# see test directory for details to set arguments.
rrt_star = RRTStar()  # or rrt_star= BiRRTStar()
rrt_star.set_vehicle(check_poly, check_res, maximum_curvature)
rrt_star.preset(start, goal, grid_map, grid_res, grid_ori, obstacle, heuristic)
rrt_star.planning(times, debug)
```
## How to install
### PyPI
```shell script
$ pip install rrts
```
### From source
```shell script
$ git clone https://github.com/liespace/pyRRTs.git
$ cd pyRRTs
$ python setup.py sdist
# install
$ pip install rrts -f dist/* --no-cache-dir
# or upload yours
$ twine upload dist/*
```
## Reference
[^1]: Karaman, Sertac, and Emilio Frazzoli. "Sampling-based algorithms for optimal motion planning." The international journal of robotics research 30.7 (2011): 846-894.

[^2]: Jordan, Matthew, and Alejandro Perez. "Optimal bidirectional rapidly-exploring random trees." (2013).
