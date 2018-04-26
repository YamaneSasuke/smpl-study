# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 20:12:58 2018

@author: yamane
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def rotation(omega):
    I = np.eye(3, dtype=omega.dtype)
    theta = np.linalg.norm(omega)
    if theta == 0:
        return np.eye(3, dtype=omega.dtype)

    omega_unit = omega / theta
    K = skew(omega_unit)
    R = I + np.sin(theta) * K + (1 - np.cos(theta)) * K.dot(K)
    return R

def transform(link, angle):
    G = np.zeros((4, 4))
    R = rotation(angle)
    G[:3, :3] = R
    G[:, 3] = np.array([link[0], link[1], link[2], 1])
    return G

def trans(x, mat):
    x = np.atleast_2d(x)
    n = len(x)
    x_hom = np.hstack((x, np.ones((n, 1))))
    return x_hom.dot(mat.T)[:, :-1]

def link(t):
    links = []
    ll = [0, 0, 0]
    for l in t:
        links.append(l-ll)
        ll = l
    return links

def skew(p):
    #  3 次元ベクトルを歪対称行列に変換
    V = np.array([[0, -p[2], p[1]],
                  [p[2], 0, -p[0]],
                  [-p[1], p[0], 0]])
    return V


if __name__ == '__main__':
    t_bar = np.array([[0, 0, 0],
                      [1, 0, 0],
                      [2, 0, 0],
                      [3, 0, 0]])
    links = np.array(link(t_bar))
    angles = np.array([[0, 0, 0],  # root
                       [2, 2, 0],  # 1
                       [0, 0, 0],  # 2
                       [0, 0, 0]])  # 3

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot(*t_bar.T, 'o-')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_zlim(-4, 4)
    plt.show()

    o = np.array([[0, 0, 0]])
    J = np.empty((0, 3))
    for l, a in zip(links[::-1], angles[::-1]):
#        omega = np.deg2rad(a)
        omega = a
        J = np.vstack((J, o))
        G = transform(l, omega)
        J = trans(J, G)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot(*J.T, 'o-')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_zlim(-4, 4)
    plt.show()