#! /usr/bin/python

import numpy as np
import rospy
import time
from numpy import genfromtxt
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
import tf
import tf2_ros
import random
from math import pi
import sys

CELL_OCC = 0.8
CELL_FREE = 0.01
CELL_UNKNOWN = 0.19
CELL_CONFLICT = 0

BASE_LINK_FRAME = 'base_link'
MAP_RESOLUTION = 0.1
MAP_SIZE_X = 50.0
MAP_SIZE_Y = 50.0
MAP_CENTER_X = -20.0
MAP_CENTER_Y = -10.0
MAP_ROWS = int(MAP_SIZE_Y / MAP_RESOLUTION)
MAP_COLS = int(MAP_SIZE_X / MAP_RESOLUTION)


class CellGrid:
    initialized = False

    def __init__(self):
        # 4 states of representation of cell
        self.grid = np.ones((MAP_ROWS, MAP_COLS, 4))
        self.cell_occ = CELL_OCC
        self.cell_free = CELL_FREE
        self.cell_unknown = CELL_UNKNOWN
        self.cell_conflict = CELL_CONFLICT

    def bayesian_approximation(self):
        for i in range(0, MAP_ROWS):
            for j in range(0, MAP_COLS):
                bayesian_constant = self.grid[i][j][0] + self.grid[i][j][1] + self.grid[i][j][2] * 2
                self.grid[i][j][0] = (self.grid[i][j][0] + self.grid[i][j][2]) / bayesian_constant
                self.grid[i][j][1] = (self.grid[i][j][1] + self.grid[i][j][2]) / bayesian_constant
                self.grid[i][j][3] = 0

    def fill_grid(self):
        for i in range(0, MAP_ROWS):
            for j in range(0, MAP_COLS):
                self.grid[i][j][0] = 0.7
                self.grid[i][j][1] = self.cell_free
                self.grid[i][j][2] = self.cell_unknown
                self.grid[i][j][3] = self.cell_conflict
        return self.grid

    def flatten_grid(self):
        # RETURN
        # self.bayesian_approximation()
        return self.grid[:, :, :-3].flatten()

    def shape(self, i):
        return self.grid.shape[i]

    def set_occ(self, i, j, mass):
        self.grid[i][j][0] = mass

    def set_free(self, i, j, mass):
        self.grid[i][j][1] = mass

    def set_unknown(self, i, j, mass):
        self.grid[i][j][2] = mass

    def set_conflict(self, i, j, mass):
        self.grid[i][j][3] = mass

    def get_cell_occ(self, i, j):
        return self.grid[i][j][0]

    def set_cell_occ(self, i, j, mass):
        self.grid[i][j][0] += mass

    def get_cell_free(self, i, j):
        return self.grid[i][j][1]

    def set_cell_free(self, i, j, mass):
        self.grid[i][j][1] += mass

    def get_cell_unknown(self, i, j):
        return self.grid[i][j][2]

    def set_cell_unknown(self, i, j, mass):
        self.grid[i][j][2] += mass

    def get_cell_conflict(self, i, j):
        return self.grid[i][j][3]

    def set_cell_conflict(self, i, j, mass):
        self.grid[i][j][3] += mass

    def get_cell_any(self, i, j):
        return 1 - self.grid[i][j][0] - self.grid[i][j][1] - self.grid[i][j][2] - self.grid[i][j][3]


class OccGrid:

    def __init__(self):
        rospy.init_node("OccupancyGrid")
        self.map_last_publish = rospy.Time()

        self.robot_frame = rospy.get_param('~robot_frame', "base_link")
        self.map_frame = rospy.get_param('~map_frame', "odom_combined")
        self.sim_time = rospy.get_param('~use_sim_time', 'true')

        self.msg = OccupancyGrid()
        self.msg.header.frame_id = self.map_frame
        self.msg.info.resolution = MAP_RESOLUTION
        self.msg.info.width = MAP_COLS
        self.msg.info.height = MAP_ROWS
        self.msg.info.origin.position.x = MAP_CENTER_X
        self.msg.info.origin.position.y = MAP_CENTER_Y

        self.grid = CellGrid()
        self.grid.fill_grid()

        self.prev_robot_x = 0
        self.prev_robot_y = 0
        self.prev_map = CellGrid()
        self.prev_map.fill_grid()

        self.sub = rospy.Subscriber('base_scan', LaserScan, self.create_grid, queue_size=1)
        self.pub = rospy.Publisher('map', OccupancyGrid, queue_size=1)

        self.tf_sub = tf.TransformListener()

        self.initial_coord_laser_scan = np.ones((MAP_ROWS, MAP_COLS))
        self.tmp_coords = np.ones((MAP_ROWS, MAP_COLS))
        self.first_update = True

    def is_inside(self, i, j):
        return i < self.grid.shape(0) and j < self.grid.shape(1) and i >= 0 and j >= 0

    def prob(self, log):
        res = np.exp(log) / (1.0 + np.exp(log))
        return res

    def log(self, prob):
        return np.log(prob / (1 - prob))

    def quarternion_to_yaw(self, qx, qy, qz, qw):
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        return np.arctan2(siny_cosp, cosy_cosp)

    def to_ij(self, x, y):
        i = (y - MAP_CENTER_Y) / MAP_RESOLUTION
        j = (x - MAP_CENTER_X) / MAP_RESOLUTION
        return i, j

    def create_grid(self, data):
        angle_min = data.angle_min
        angle_inc = data.angle_increment

        now = rospy.Time(0)
        self.tf_sub.waitForTransform(self.map_frame, self.robot_frame, now, rospy.Duration(1.0))
        try:
            (x, y, _), (qx, qy, qz, qw) = self.tf_sub.lookupTransform(self.map_frame, self.robot_frame, now)
            theta = self.quarternion_to_yaw(qx, qy, qz, qw)

            # if update is needed
            if (x - self.prev_robot_x) ** 2 + (y - self.prev_robot_y) ** 2 >= 0.01 ** 2:
                # grid = self.update_map(x, y, theta, data)
                # grid.bayesian_approximation()
                occ_grid = self.update_map(x, y, theta, data).flatten_grid()  # update map
                # print(occ_grid)
                # self.matching(prev_occ_grid, occ_grid)

                self.prev_robot_x = x
                self.prev_robot_y = y

                if self.map_last_publish.to_sec() + 1.0 < rospy.Time.now().to_sec():
                    self.map_last_publish = rospy.Time.now()

                    # occ_grid = self.matching_simple().flatten_grid()

                    self.publish_grid(occ_grid, data.header.stamp)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(e)

    def update_map(self, x, y, theta, data):
        if self.first_update:
            # print('Once initialize')
            for idx, value in enumerate(data.ranges):
                self.update_cells(x, y, (theta + data.angle_min + idx * data.angle_increment), value)
            self.first_update = False
        else:
            # print('Add new laser scan')
            # arr = np.array((x, y))
            x0, y0 = self.to_ij(x, y)
            list_coord = [[x0, y0, (theta + data.angle_min), 0]]
            for idx, value in enumerate(data.ranges):
                # if idx == 0:
                #     x0, y0 = self.to_ij(x, y)
                #     list_coord.append([x0, y0, (theta + data.angle_min), value])
                # else:
                self.save_occ_coords(x, y, (theta + data.angle_min + idx * data.angle_increment), value, list_coord)
            # self.add_to_map_new_coord(list_coord)
            list_best_coord = self.scan_matcher(list_coord)
            self.add_to_map_new_coord(list_best_coord)
        return self.grid

    def generate_pose(self, list_coord):
        shift_transl = random.uniform(-10.0, 10.0)
        shift_rotation = random.uniform(-pi, pi)
        new_list_coord = []

        for i in range(0, len(list_coord)):
            new_x = list_coord[i][0] + shift_transl
            new_y = list_coord[i][1] + shift_transl
            new_theta = shift_rotation
            new_list_coord.append([new_x, new_y, new_theta, list_coord[i][3]])
        return new_list_coord

    def scan_matcher(self, list_coord):
        max_score = 0
        min_score = 0
        best_coord = []
        for k in range(0, 300):
            new_list_coord = self.generate_pose(list_coord)
            # self.add_to_map_new_coord(new_list_coord)
            score = 0
            for i in range(0, len(new_list_coord)):
                if self.grid.get_cell_occ(int(new_list_coord[i][0]), int(new_list_coord[i][1])) == CELL_OCC:
                    score = score + 1
                else:
                    score = score - 1
            if score > max_score:
                max_score = score
                best_coord = new_list_coord
            else:
                if score > min_score:
                    min_score = score
                    best_coord = new_list_coord
        if max_score == 0 and min_score != len(list_coord):
            return best_coord
        else:
            if max_score != 0:
                return best_coord
            else:
                return list_coord

    def add_to_map_new_coord(self, new_coord):
        for i in range(1, len(new_coord)):
            if self.grid.get_cell_occ(int(new_coord[i][0]), int(new_coord[i][1])) != CELL_OCC:
                ip, jp = self.bresenham(new_coord[0][0], new_coord[0][1], new_coord[i][0], new_coord[i][1],
                                        new_coord[i][3] / MAP_RESOLUTION)
                self.grid.set_occ(int(ip), int(jp), CELL_OCC)
                self.grid.set_free(int(ip), int(jp), CELL_FREE)
                self.grid.set_unknown(int(ip), int(jp), CELL_UNKNOWN)
                self.grid.set_conflict(int(ip), int(jp), CELL_CONFLICT)

    def save_occ_coords(self, x0, y0, theta, value, list_coord):
        x1 = x0 + value * np.cos(theta)
        y1 = y0 + value * np.sin(theta)

        # into map coordinates
        # i0, j0 = self.to_ij(x0, y0)
        i1, j1 = self.to_ij(x1, y1)
        list_coord.append([i1, j1, theta, value])
        # self.tmp_coords[i1][j1] = CELL_OCC

    def update_cells(self, x0, y0, theta, value):
        x1 = x0 + value * np.cos(theta)
        y1 = y0 + value * np.sin(theta)

        # into map coordinates
        i0, j0 = self.to_ij(x0, y0)
        i1, j1 = self.to_ij(x1, y1)
        # occupied cells
        # self.combination(int(ip), int(jp), False)

        # if self.first_update:
        d_cells = value / MAP_RESOLUTION
        ip, jp = self.bresenham(i0, j0, i1, j1, d_cells)
        self.grid.set_occ(int(ip), int(jp), CELL_OCC)
        self.grid.set_free(int(ip), int(jp), CELL_FREE)
        self.grid.set_unknown(int(ip), int(jp), CELL_UNKNOWN)
        self.grid.set_conflict(int(ip), int(jp), CELL_CONFLICT)

    def bresenham(self, x0, y0, x1, y1, d):
        dx = abs(y1 - y0)
        sx = 1 if y0 < y1 else -1

        dy = -1 * abs(x1 - x0)
        sy = 1 if x0 < x1 else -1

        jp, ip = y0, x0
        err = dx + dy

        while True:
            if (jp == y1 and ip == x1) or (np.sqrt((jp - y0) ** 2 + (ip - x0) ** 2) >= d) or not self.is_inside(ip, jp):
                return ip, jp
            elif self.grid.get_cell_occ(int(ip), int(jp)) == 100:
                return ip, jp

            # unoccupied cells
            if self.is_inside(ip, jp):
                # self.combination(int(ip), int(jp), True)
                self.grid.set_occ(int(ip), int(jp), CELL_FREE)
                self.grid.set_free(int(ip), int(jp), CELL_OCC)
                self.grid.set_unknown(int(ip), int(jp), CELL_UNKNOWN)
                self.grid.set_conflict(int(ip), int(jp), CELL_CONFLICT)

            e2 = 2 * err
            if e2 >= dy:
                err += dy
                jp += sx
            if e2 <= dx:
                err += dx
                ip += sy

    def combination_rule(self, i, j, is_inside):
        prev_occ = self.grid.get_cell_occ(i, j)
        prev_free = self.grid.get_cell_free(i, j)
        prev_unk = self.grid.get_cell_unknown(i, j)
        prev_conflict = 0

        if (is_inside):
            curr_occ = CELL_FREE
            curr_free = CELL_OCC
        else:
            curr_occ = CELL_OCC
            curr_free = CELL_FREE
        curr_unk = CELL_UNKNOWN
        curr_conflict = 0

        new_occ = prev_occ * curr_occ + prev_occ * curr_unk + prev_unk * curr_occ  # 0,944
        new_free = prev_free * curr_free + prev_free * curr_unk + prev_unk * curr_free  # 0,0039
        new_unk = prev_unk * curr_unk  # 0.0361
        new_conflict = 0
        empty = prev_occ * curr_free + prev_free * curr_occ
        koeff = 1 - empty  # 0.984

        self.grid.set_occ(i, j, new_occ / koeff)
        self.grid.set_free(i, j, new_free / koeff)
        self.grid.set_unknown(i, j, new_unk / koeff)
        self.grid.set_conflict(i, j, new_conflict / koeff)

    def publish_grid(self, occ_grid, stamp):
        probability_map = (occ_grid * 100).astype(dtype=np.int8)
        self.msg.data = probability_map
        self.msg.header.stamp = stamp
        self.pub.publish(self.msg)
        rospy.loginfo("!")

occupancy_grid = OccGrid()
while not rospy.is_shutdown():
    rospy.spin()
