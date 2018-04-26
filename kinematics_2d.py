# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 16:48:04 2018

@author: yamane
"""

import numpy as np
import matplotlib.pyplot as plt


def rotation(degree):
    rad = np.deg2rad(degree)
    cos = np.cos(rad)
    sin = np.sin(rad)
    return np.array([[cos, -sin],
                     [sin, cos]])


def transform(link, angle):
    G = np.zeros((3, 3))
    R = rotation(angle)
    G[:2, :2] = R
    G[:, 2] = np.array([link[0], link[1], 1])
    return G


def trans(x, mat):
    x = np.atleast_2d(x)
    n = len(x)
    x_hom = np.hstack((x, np.ones((n, 1))))
    return x_hom.dot(mat.T)[:, :-1]


def link(t):
    links = []
    ll = [0, 0]
    for l in t:
        links.append(l-ll)
        ll = l
    return links


if __name__ == '__main__':
    t_bar = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
    links = np.array(link(t_bar))
    angles = np.array([45, -45, 0, 0])

    plt.plot(*t_bar.T, 'o-')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.grid()
    plt.show()

    o = np.array([[0, 0]])
    J = np.empty((0, 2))
    for l, a in zip(links[::-1], angles[::-1]):
        J = np.vstack((J, o))
        G = transform(l, a)
        J = trans(J, G)
        plt.plot(*J.T, 'o-')
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.grid()
        plt.show()