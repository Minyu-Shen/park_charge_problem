import random
import math


def Expntl(L, n):
    """
    以L作为参数产生n个负指数分布随机数，到达间隔
    negative exponential distribution
    return n double random number, L is the mean value
    """
    arrival_interval = [0] * n
    for i in range(0, n):
        tep = random.random()
        arrival_interval[i] = -L * math.log(tep)
    return arrival_interval


def cal_G(rewards, gamma):
    Gs = []
    discounted_reward = 0
    for rew in reversed(rewards):
        discounted_reward = rew + discounted_reward * gamma
        Gs.insert(0, discounted_reward)
    assert len(Gs) == len(rewards), "mush have same shape"
    return Gs


# print(cal_G([1, 2, 3], 1))

