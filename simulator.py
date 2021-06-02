# -*- coding: utf-8 -*-
"""
Created on Thu May 27 08:44:22 2021

@author: DELL
"""
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 16:41:13 2021

@author: RuiJie&&MinYu Parking and Charging Problem
"""
import numpy as np
import math
import copy
import random
from arena import Expntl

np.random.seed(0)
random.seed(0)


class Simulator(object):
    def __init__(self):
        self.parameter_lambda = 5
        self.customer_number = 1000
        self.time_peroid = 60 * 12  ##### arrival  time period
        self.total_peroid = 60 * 16
        self.parking_region = 2
        self.region_parking_lot = 3
        self.region_charging_gun = 2
        self.parking_price = 0.03
        self.charging_price = 0.02
        self.parking_time_para = 300
        self.curr_arrived_idx = -1
        self.pre_state = []
        self.post_state = []
        self.customer_arrival_time = []
        self.customer_parking_time = []
        self.customer_charging_time = []
        self.customer_price = []
        self.curr_arrived_info = [0, 0, 0]
        self.reward = 0

    ########################################################
    def reset(self):
        self._generate_demand()
        self.pre_state = np.zeros(
            2 * self.parking_region * self.region_parking_lot + 1
        )  # initial state
        self.post_state = np.zeros(
            2 * self.parking_region * self.region_parking_lot + 1
        )  # initial state
        self.feasible_parking_desicion = np.zeros(self.parking_region)
        self.parking_desicion = np.zeros(self.parking_region)
        self.action = np.zeros(self.parking_region)

        return (
            copy.deepcopy(self.pre_state),
            copy.deepcopy(self.curr_arrived_info),
        )  # ? should be the first arrival info

    ######################################################################
    def _generate_demand(self):
        """
            for each episode, generate demand realization
            """
        self._generate_customer_arrival_time()
        self._truncate_arrival_within_period()
        self._generate_customer_parking_time()
        self._generate_customer_charging_time()
        self._generate_customer_price()

    ######################################################################
    def _generate_customer_arrival_time(self):
        """
            生成泊松过程：乘客到达时间节点
            """
        arrival_interval = Expntl(self.parameter_lambda, self.customer_number)
        for i in range(0, len(arrival_interval)):
            self.customer_arrival_time.append(sum(arrival_interval[0 : (i + 1)]))

    ######################################################################
    def _truncate_arrival_within_period(self):
        """
            选取[0,time_peroid]的顾客，丢掉其他顾客数据
            """
        self.customer_arrival_time = np.array(self.customer_arrival_time)
        index = np.where(self.customer_arrival_time > self.time_peroid)
        first_index = index[0][0]
        self.customer_arrival_time = self.customer_arrival_time[0:first_index]

    ######################################################################
    def _generate_customer_parking_time(self):
        """
        随机生成顾客停车需求
        """
        start_parking = self.customer_arrival_time
        parking_time = np.random.exponential(
            self.parking_time_para, len(self.customer_arrival_time)
        )
        end_parking = start_parking + parking_time
        self.customer_parking_time = np.array((start_parking, end_parking))

    ######################################################################

    def _generate_customer_charging_time(self):
        """
            随机生成顾客充电需求
            """
        # customer_number = self.customer_parking_time.shape[1]
        for i in range(0, len(self.customer_arrival_time)):
            charging_time = random.uniform(
                0, self.customer_parking_time[1][i] - self.customer_parking_time[0][i]
            )
            self.customer_charging_time.append(charging_time)
        # self.customer_charging_time = np.array(self.customer_charging_time)

    ######################################################################

    def _generate_customer_price(self):
        """
            价格
            """
        parking_time_len = (
            self.customer_parking_time[1][:] - self.customer_parking_time[0][:] + 1
        )
        customer_parking_price = self.parking_price * parking_time_len
        self.customer_charging_time = np.asarray(self.customer_charging_time)
        customer_charging_price = self.charging_price * self.customer_charging_time
        self.customer_price = customer_parking_price + customer_charging_price

    ######################################################################
    def _update_i_arrival_info(self):
        self.curr_arrived_info = [
            self.customer_parking_time[0][self.curr_arrived_idx],
            self.customer_parking_time[1][self.curr_arrived_idx],
            self.customer_charging_time[self.curr_arrived_idx],
        ]

    ######################################################################
    def step(self, step, action):

        self.curr_arrived_idx = step

        self._update_i_arrival_info()
        # print("arrival_info", self.curr_arrived_info)
        if step == 0:
            self.pre_state  ### ???
        else:
            self._updata_pre_state()
        # print('pre_state', self.pre_state)

        self._feasible_parking_desicion()
        # print('feasible_parking_desicion', self.feasible_parking_desicion)

        # self._action() ### ???
        self.action = action

        # print('action', self.action)

        self._updata_post_state()
        # print('post_state', self.post_state)
        self._reward()
        # print('reward', self.reward)

        is_done = (
            True
            if self.curr_arrived_idx >= len(self.customer_arrival_time) - 1
            else False
        )

        return (
            copy.deepcopy(self.pre_state),
            copy.deepcopy(self.curr_arrived_info),
            # copy.deepcopy(self.action),
            copy.deepcopy(self.reward),
            is_done,
        )

    ######################################################################
    def _reward(self):
        if sum(self.action) == 0:
            self.reward = 0
        else:
            self.reward = self.customer_price[self.curr_arrived_idx]

    ######################################################################

    def _updata_post_state(self):
        self.post_state = self.pre_state
        self.post_state[-1] = self.curr_arrived_info[0]
        ## 如果接受了request,更新状态
        if sum(self.action) > 0:
            index = np.where(self.action == 1)
            i_parking_region = index[0][0]
            ## 更新parking 离开时间
            sub_post_state = self.post_state[
                i_parking_region
                * self.region_parking_lot : (i_parking_region + 1)
                * self.region_parking_lot
            ]

            index = np.where(sub_post_state == 0)
            index1 = i_parking_region * self.region_parking_lot + index[0][0]
            self.post_state[index1] = (
                self.curr_arrived_info[1] - self.curr_arrived_info[0]
            )
            ## 更新charging demand
            index2 = index1 + self.parking_region * self.region_parking_lot
            self.post_state[index2] = self.curr_arrived_info[2]

    ######################################################################

    def _action(self):
        if sum(self.feasible_parking_desicion) > 0:
            index = np.where(np.array(self.feasible_parking_desicion) == 1)
            self.action = np.array([0] * len(self.feasible_parking_desicion))
            parking_region_index = index[0][0]
            self.action[parking_region_index] = 1
        else:
            self.action = np.array([0] * len(self.feasible_parking_desicion))

    ######################################################################

    def _updata_pre_state(self):
        # 更新到达时间节点
        self.pre_state[-1] = self.curr_arrived_info[0]
        # 更新充电状态
        self._updata_pre_charging_state()
        # 更新是否离开以及还有多久离开
        self._updata_pre_parking_state()
        # print('a', self.pre_state)

    ################################################
    #    state: 2*parking_region*region_parking_lot+1 列，
    #    前parking_region*region_parking维度存储离开时间，
    #    后parking_region*region_parking维度存储剩余充电时间，
    #    最后一个维度存储当前时间节点
    ###################
    def _updata_pre_charging_state(self):
        i = self.curr_arrived_idx
        arrival_time = self.customer_arrival_time[i]
        pre_arrival_time = self.customer_arrival_time[i - 1]
        available_charging_demand = [
            arrival_time - pre_arrival_time
        ] * self.region_charging_gun
        ### 停车的顾客
        sub_pre_state = self.pre_state[
            0 : self.parking_region * self.region_parking_lot
        ]
        index = np.where(sub_pre_state > 0)[0]
        departure_time = self.pre_state[index]
        ## tep 两行，第一行是位置index,第二行是离开时间
        tep = np.vstack((index, departure_time))
        tep = tep[:, np.argsort(tep[1, :])]
        if index.size > 0:
            for k in range(0, len(index)):
                available_charging_demand.sort(reverse=True)
                k_charging_index = int(
                    tep[0][k] + self.parking_region * self.region_parking_lot
                )
                k_charging_demand = self.pre_state[k_charging_index]
                if k_charging_demand >= available_charging_demand[0]:
                    self.pre_state[k_charging_index] = (
                        k_charging_demand - available_charging_demand[0]
                    )
                    available_charging_demand[0] = 0
                else:
                    self.pre_state[k_charging_index] = 0
                    available_charging_demand[0] = (
                        available_charging_demand[0] - k_charging_demand
                    )

    ################################################
    #    state: 2*parking_region*region_parking_lot+1 列，
    #    前parking_region*region_parking维度存储离开时间，
    #    后parking_region*region_parking维度存储剩余充电时间，
    #    最后一个维度存储当前时间节点
    ###################
    def _updata_pre_parking_state(self):
        i = self.curr_arrived_idx
        arrival_time = self.customer_arrival_time[i]

        sub_pre_state = self.pre_state[
            0 : self.parking_region * self.region_parking_lot
        ]
        index = np.where(sub_pre_state > 0)[0]
        departure_time = self.pre_state[index]
        ## tep 是两行，第一行是index, 第二行是 departure_time
        tep = np.vstack((index, departure_time))
        # tep = tep[:,np.argsort(tep[1,:])]
        if index.size > 0:
            for k in range(0, len(index)):
                k_departure_time = tep[1][k]
                if k_departure_time <= arrival_time:
                    ## parking 状态设为 0
                    departure_index = int(tep[0][k])
                    self.pre_state[departure_index] = 0
                    ## charging 状态设为 0
                    charging_index = (
                        departure_index + self.parking_region * self.region_parking_lot
                    )
                    self.pre_state[charging_index] = 0

        ##
        sub_pre_state = self.pre_state[
            0 : self.parking_region * self.region_parking_lot
        ]
        index = np.where(sub_pre_state > 0)[0]
        tep = self.customer_arrival_time[i] - self.customer_arrival_time[i - 1]
        self.pre_state[index] = self.pre_state[index] - tep

    ######################################
    def _feasible_parking_desicion(self):
        """
        首先不考虑充电需求，根据停车位占用情况，返回可行的feasible parking region, 如果可行，则令 1，否则为0
        然后考虑充电需求，，返回返回可行的feasible parking region, 如果可行，则令 1，否则为0
        """
        self.parking_space_feasible()
        self.charging_feasible()

    ######################################
    def parking_space_feasible(self):
        for i in range(0, self.parking_region):
            index1 = i * self.region_parking_lot
            index2 = (i + 1) * self.region_parking_lot
            sub_pre_state = self.pre_state[index1:index2]
            index = np.where(sub_pre_state > 0)[0]
            if index.size < self.region_parking_lot:
                self.feasible_parking_desicion[i] = 1
            else:
                self.feasible_parking_desicion[i] = 0

    ######################################
    #    state: 2*parking_region*region_parking_lot+1 列，
    #    前parking_region*region_parking维度存储离开时间，
    #    后parking_region*region_parking维度存储剩余充电时间，
    #    最后一个维度存储当前时间节点
    def charging_feasible(self):
        for k in range(0, self.parking_region):
            ### 如果parking_space fessible,才检查charging 是否feassible.
            if self.feasible_parking_desicion[k] == 1:
                index1 = k * self.region_parking_lot
                index2 = (k + 1) * self.region_parking_lot
                sub_pre_state = self.pre_state[index1:index2]

                index = np.where(sub_pre_state > 0)[0]
                departure_time = self.pre_state[index]
                charging_time = self.pre_state[
                    index + self.parking_region * self.region_parking_lot
                ]
                ## tep 是两行，第一行是departure_time, 第二行是剩余 charging_time
                tep = np.vstack((departure_time, charging_time))
                tep1 = np.array([self.curr_arrived_info[1], self.curr_arrived_info[2]])
                tep = np.c_[tep, tep1]
                ## 按照第一行（离开时间）升序排列
                tep = tep[:, np.argsort(tep[0, :])]
                ## 可用充电
                arrival_time = self.curr_arrived_info[0]
                max_departure_time = max(tep[0])
                available_charging_demand = [
                    max_departure_time - arrival_time
                ] * self.region_charging_gun
                ##
                for j in range(0, tep[0].size):
                    available_charging_demand.sort(reverse=True)
                    if available_charging_demand[0] < tep[1][j]:
                        self.feasible_parking_desicion[k] = 0
                        break
                    else:
                        available_charging_demand[0] = (
                            available_charging_demand[0] - tep[1][j]
                        )

