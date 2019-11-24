# -*- coding: utf-8 -*-
"""All the transformations required for the SubgridTree class.
Clint Lombard
"""

import numpy as np


# Define transform functions
def __t1(x):
    return 2 * x


def __t2(x):
    return 2 * x - 1


def __t3(x):
    return 1 - 2 * x


def tri0_trans(p):
    return np.array([__t2(p[0]), __t1(p[1]), __t1(p[2])])


def tri1_trans(p):
    return np.array([__t1(p[0]), __t2(p[1]), __t1(p[2])])


def tri2_trans(p):
    return np.array([__t1(p[0]), __t1(p[1]), __t2(p[2])])


def tri3_trans(p):
    return np.array([__t3(p[0]), __t3(p[1]), __t3(p[2])])


# Define inverse transform functions
def __t1_inv(x):
    return 0.5 * x


def __t2_inv(x):
    return 0.5 * (x + 1)


def __t3_inv(x):
    return 0.5 * (1 - x)


def tri0_trans_inv(p):
    return np.array([__t2_inv(p[0]), __t1_inv(p[1]), __t1_inv(p[2])])


def tri1_trans_inv(p):
    return np.array([__t1_inv(p[0]), __t2_inv(p[1]), __t1_inv(p[2])])


def tri2_trans_inv(p):
    return np.array([__t1_inv(p[0]), __t1_inv(p[1]), __t2_inv(p[2])])


def tri3_trans_inv(p):
    return np.array([__t3_inv(p[0]), __t3_inv(p[1]), __t3_inv(p[2])])
