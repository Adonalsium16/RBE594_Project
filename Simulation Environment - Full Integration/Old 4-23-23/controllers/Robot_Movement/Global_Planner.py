from a_star_searching_from_two_side import *
import math
import csv
import numpy as np
import matplotlib.pyplot as plt

def wrap(index, l):
    """ Helper function to get valid index of list
    """
    if index >= len(l):
        return len(l) - 1
    elif index < 0:
        return 0
    else:
        return index
    
class HistogramGrid:
    """ Class HistogramGrid defines a nested array ("grid") of certainty values
        Coordinate points start from 0
    """

    MAX_CERTAINTY = 15

    def __init__(self, nrows, ncols):
        self.grid = [([0] * ncols)] * nrows

    def out_of_bounds(self, x, y):
        """ Returns whether the cell is out of the grid. Used for edge conditions """
        return 0 > y or y >= len(self.grid) or 0 > x or x >= len(self.grid[0])

    def get_valid_x(self, index):
        return wrap(index, self.grid[0])

    def get_valid_y(self, index):
        return wrap(index, self.grid)

    def get_certainty(self, x, y):
        return self.grid[y][x]

    def add_certainty(self, x, y):
        """ Increments cell certainty by one, capped at MAX_CERTAINTY.
            Maybe change increment value to parameter in future iterations """
        if self.grid[y][x] < self.MAX_CERTAINTY:
            self.grid[y][x] += 1

    def print_hg(self, robot_locations, start, end, current):
        """ For testing purposes """
        string = ""
        for y in range(len(self.grid)):
            for x in range(len(self.grid[0])):

                if self.get_certainty(x, y) == 1:
                    string += "1 "
                elif (x, y) == start:
                    string += "S "
                elif (x, y) == end:
                    string += "E "
                elif (x, y) == current:
                    string += "C "
                elif (x, y) in robot_locations:
                    string += "X "
                else:
                    string += "0 "
            string += "\n"
        string += "0/1 - Free/Occupied (Certainty values)\nX - Robot locations\nS - Start Position (%d, %d)\nE - End Target (%d, %d)\nC - Current" % (
            start[0], start[1], end[0], end[1])
        print (string)

def from_map(map_fname):
        """ Create grid from text file """
        with open(map_fname, 'r') as f:
            reader = csv.reader(f, delimiter=" ")
            lines = list(reader)
            lines.reverse()

        for l in lines:
            if '' in l:
                l.remove('')
            
        lines = list(map(lambda l: list(map(int, l)), lines))
        return lines
    
def obstacle_grid(grid_map,start,end):
    hg_dim = (50, 50)
    
    hg = HistogramGrid(hg_dim[0], hg_dim[1])
    hg.grid = from_map(grid_map)
    
    top_vertex = [len(hg.grid)-1, len(hg.grid[0])-1]  # top right vertex of boundary
    bottom_vertex = [0, 0]  # bottom left vertex of boundary
        
    ay = list(range(bottom_vertex[1], top_vertex[1]))
    ax = [bottom_vertex[0]] * len(ay)
    cy = ay
    cx = [top_vertex[0]] * len(cy)
    bx = list(range(bottom_vertex[0] + 1, top_vertex[0]))
    by = [bottom_vertex[1]] * len(bx)
    dx = [bottom_vertex[0]] + bx + [top_vertex[0]]
    dy = [top_vertex[1]] * len(dx)
    
    # x y coordinate in certain order for boundary
    x = ax + bx + cx + dx
    y = ay + by + cy + dy
    
    
    i = 1
    ob_x = []
    ob_y = []
    while i < len(hg.grid)-1:
        j = 1
        while j < len(hg.grid[0])-1:
            if hg.grid[i][j] == 1:
                ob_x.append(j)
                ob_y.append(i)
            j += 1
        i += 1
            
    obstacle = np.vstack((ob_x, ob_y)).T.tolist()
    # remove start and end coordinate in obstacle list
    obstacle = [coor for coor in obstacle if coor != start and coor != end]
    obs_array = np.array(obstacle)
    bound_temp = np.vstack((x, y)).T
    bound = np.vstack((bound_temp, obs_array))

    return bound,obstacle

def global_planner():
    start = [[72,67],[22,88],[23,46],[68,48],[67,2]]
    end = [[72,67],[22,88],[23,46],[68,48],[67,2]]

    #start = [[67,72],[88,22],[46,23],[48,68],[2,67]]
    #end = [[67,72],[88,22],[46,23],[48,68],[2,67]]

    start7267 = []
    start2288 = []
    start2346 = []
    start6848 = []
    start6702 = []

    """ Testing for A* algorithm using gazebo_map.txt grid"""
    for i in range(len(start)):
        for j in range(len(end)):
            if start[i]==end[j]:
                pass
            else:
                bound,obstacle = obstacle_grid("gazebo_map.txt",start[i],end[j])
                path = searching_control(start[i], end[j], bound, obstacle)
                if start[i] == start[0]:
                    start7267.append(path)
                elif start[i] == start[1]:
                    start2288.append(path)
                elif start[i] == start[2]:
                    start2346.append(path)
                elif start[i] == start[3]:
                    start6848.append(path)
                elif start[i] == start[4]:
                    start6702.append(path)

                markersize = 5.1
                plt.plot(path[:, 0], path[:, 1], 'or', markersize=markersize/2, alpha = 0.2)
                plt.plot(bound[:, 0], bound[:, 1], 'sk', markersize=markersize)
                plt.plot(end[j][0], end[j][1], '*b', label='Goal')
                plt.plot(start[i][0], start[i][1], '^b', label='Origin')

    return start7267,start2288,start2346,start6848,start6702

#a,b,c,d,e = global_planner()

#print(a)