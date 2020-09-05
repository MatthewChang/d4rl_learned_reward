import numpy as np
import cv2
from matplotlib import pyplot as plt 

WALL = 10
EMPTY = 11
GOAL = 12
START = 13

def parse_maze(maze_str):
    lines = maze_str.strip().split('\\')
    width, height = len(lines), len(lines[0])
    maze_arr = np.zeros((width, height), dtype=np.int32)
    for w in range(width):
        for h in range(height):
            tile = lines[w][h]
            if tile == '#':
                maze_arr[w][h] = WALL
            elif tile == 'G':
                maze_arr[w][h] = GOAL
            elif tile == ' ' or tile == 'O' or tile == '0':
                maze_arr[w][h] = EMPTY
            elif tile == 'S':
                maze_arr[w][h] = START
            else:
                raise ValueError('Unknown tile type: %s' % tile)
    return maze_arr

LARGE_MAZE = \
        "############\\"+\
        "#OOOO#OOOOS#\\"+\
        "#O##O#O#O#O#\\"+\
        "#OOOOOO#OOO#\\"+\
        "#O####O###O#\\"+\
        "#OO#O#OOOOO#\\"+\
        "##O#O#O#O###\\"+\
        "#GO#OOO#OOO#\\"+\
        "############"

parsed = parse_maze(LARGE_MAZE)
width,height = parsed.shape
scale = 100
m = np.zeros((width*scale,height*scale))
parsed[parsed==WALL] = 1
parsed[parsed!=1] = 0 
parsed

scaled = cv2.resize(parsed,None,fx=100,fy=100,interpolation=cv2.INTER_NEAREST)
plt.imsave('test.png',scaled)

import scipy, skfmm
