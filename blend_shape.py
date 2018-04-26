# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 21:16:31 2018

@author: ryuhei
"""

import numpy as np
import matplotlib.pyplot as plt

#import chainer
#import chainer.functions as F
#import chainer.links as L
#from chainer import cuda, Variable


def angle_to_rotmat(degree):
    rad = np.deg2rad(degree)
    cos = np.cos(rad)
    sin = np.sin(rad)
    return np.array([[cos, -sin],
                     [sin, cos]])


def transmat(link, angle):
    G = np.zeros((3, 3))
    R = angle_to_rotmat(angle)
    G[:2, :2] = R
    G[:, 2] = np.array([link, 0, 1])
    return G


def G_func(omega, j):
    G = np.zeros((3, 3))
    R = angle_to_rotmat(omega)
    G[:2, :2] = R
    G[:2, 2] = j
    G[2, 2] = 1
    return G


def trans(x, mat):
    x = np.atleast_2d(x)
    n = len(x)
    x_hom = np.hstack((x, np.ones((n, 1))))
    return x_hom.dot(mat.T)[:, :-1]


def create_joints_and_vertices(verbose=False):
    # Parameters of a robot arm with 3 joints, from the root to the end.
    links = np.array([0, 1, 2, 3])
    angles = np.array([60, -30, -30, 0])
    interval = 0.2
    thickness = 0.4

    K = len(links)
    o = np.array([[0, 0]])
    J = np.empty((0, 2))
    Gs = []
    for l, a in zip(links[::-1], angles[::-1]):
        J = np.vstack((J, o))
        G = transmat(l, a)
        J = trans(J, G)

        Gs.append(G)

        if verbose:
            plt.plot(*J.T, 'o-')
            plt.xlim(-7, 7)
            plt.ylim(-7, 7)
            plt.gca().set_aspect(1)
            plt.grid()
            plt.show()
    Gs = Gs[::-1]
    J = J[::-1]

    k = K - 1
    T = np.empty((0, 2))
    v = np.empty((0, 2))
    vertices_segments = []
    vx = np.arange(0, 1, interval).reshape(-1, 1)
    vy = np.linspace(thickness, 0, len(vx), False).reshape(-1, 1)
    v = np.vstack((v, np.hstack((vx, vy))))
    vy = np.linspace(-thickness, 0, len(vx), False).reshape(-1, 1)
    v = np.vstack((v, np.hstack((vx, vy))))
    for l, G in zip(links[::-1], Gs[::-1]):
        vertices_segments += [k] * len(v)
        k -= 1
        # create vertices
        T = np.vstack((T, v))
        T = trans(T, G)
        v = np.empty((0, 2))
        vx = np.arange(0, l + interval, interval).reshape(-1, 1)
        vy = np.full_like(vx, thickness)
        v = np.vstack((v, np.hstack((vx, vy))))
        vy = np.full_like(vx[1:-1], -thickness)
        v = np.vstack((v, np.hstack((vx[1:-1], vy))))

        if verbose:
            plt.plot(*J.T, 'o-')
            plt.plot(*T.T, '.')
            plt.xlim(-7, 7)
            plt.ylim(-7, 7)
            plt.gca().set_aspect(1)
            plt.grid()
            plt.show()

    N = len(vertices_segments)
    W = np.zeros((N, K), np.float32)
    W[range(N), vertices_segments] = 1.0

    return J, T, angles, W



if __name__ == '__main__':
    omega_desires = np.array([0, 30, 0, 0])
    J, T_bar, omega_stars, W = create_joints_and_vertices(False)

    plt.plot(*J.T, 'o-')
    plt.plot(*T_bar.T, '.')
    plt.xlim(-7, 7)
    plt.ylim(-7, 7)
    plt.gca().set_aspect(1)
    plt.grid()
    plt.show()

    Ts = []
    G = np.eye(3)
    G_star_inv = np.eye(3)
    G_des = np.eye(3)
    for omega_star, j, omega_des in zip(omega_stars, J, omega_desires):
        j = trans(j, G_star_inv)
        G_star_k = G_func(0, j)
        G_des_k = G_func(omega_des, j)
        G_star_inv = np.linalg.inv(G_star_k).dot(G_star_inv)
        G_des = G_des.dot(G_des_k)
        G = G_des.dot(G_star_inv)
        T_k = trans(T_bar, G)
        print(T_k.shape)
        print()
        Ts.append(T_k)

    for t in Ts:
        plt.plot(*J.T, 'o-')
        plt.plot(*t.T, '.')
        plt.xlim(-7, 7)
        plt.ylim(-7, 7)
        plt.gca().set_aspect(1)
        plt.grid()
        plt.show()

    # use einsum instead of for loop
    T_prime = np.zeros_like(T_bar)
    for t, w in zip(Ts, W.T):

        T_prime += t * w.reshape(-1, 1)

    plt.plot(*J.T, 'o-')
    plt.plot(*T_bar.T, '.')
    plt.xlim(-7, 7)
    plt.ylim(-7, 7)
    plt.gca().set_aspect(1)
    plt.grid()
    plt.show()

    plt.plot(*J.T, 'o-')
    plt.plot(*T_prime.T, '.')
    plt.xlim(-7, 7)
    plt.ylim(-7, 7)
    plt.gca().set_aspect(1)
    plt.grid()
    plt.show()