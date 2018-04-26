# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 15:48:51 2018

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

def create_joints_and_vertices(verbose=False):
    # Parameters of a robot arm with 3 joints, from the root to the end.
    t_bar = np.array([[0, 0, 0],
                      [1, 0, 0],
                      [2, 0, 0],
                      [3, 0, 0]])
    links = np.array(link(t_bar))
    angles = np.array([[0, 0, 0],  # root
                       [0, 0, 0],  # 1
                       [0, 0, 0],  # 2
                       [0, 0, 0]])  # 3
    interval = 0.2
    thickness = 0.4

    K = len(links)
    o = np.array([[0, 0, 0]])
    J = np.empty((0, 3))
    Gs = []
    for l, a in zip(links[::-1], angles[::-1]):
#        omega = np.deg2rad(a)
        omega = a
        J = np.vstack((J, o))
        G = transform(l, omega)
        J = trans(J, G)

        Gs.append(G)
    Gs = Gs[::-1]
    J = J[::-1]

    k = K - 1
    T = np.empty((0, 3))
    v = np.empty((0, 3))
    vertices_segments = []
    vx = np.arange(0, 1, interval).reshape(-1, 1)
    vy = np.linspace(thickness, 0, len(vx), False).reshape(-1, 1)
    vz = np.zeros_like(vx)
    v = np.vstack((v, np.hstack((vx, vy, vz))))
    vy = np.linspace(-thickness, 0, len(vx), False).reshape(-1, 1)
    vz = np.zeros_like(vx)
    v = np.vstack((v, np.hstack((vx, vy, vz))))

    vx = np.arange(0, 1-0.2, interval).reshape(-1, 1)
    vy = np.linspace(thickness-0.1, 0, len(vx), False).reshape(-1, 1)
    vz = np.zeros_like(vx)+0.1
    v = np.vstack((v, np.hstack((vx, vy, vz))))
    vy = np.linspace(-thickness+0.1, 0, len(vx), False).reshape(-1, 1)
    vz = np.zeros_like(vx)+0.1
    v = np.vstack((v, np.hstack((vx, vy, vz))))

    vx = np.arange(0, 1-0.4, interval).reshape(-1, 1)
    vy = np.linspace(thickness-0.2, 0, len(vx), False).reshape(-1, 1)
    vz = np.zeros_like(vx)+0.2
    v = np.vstack((v, np.hstack((vx, vy, vz))))
    vy = np.linspace(-thickness+0.2, 0, len(vx), False).reshape(-1, 1)
    vz = np.zeros_like(vx)+0.2
    v = np.vstack((v, np.hstack((vx, vy, vz))))

    vx = np.arange(0, 1-0.6, interval).reshape(-1, 1)
    vy = np.linspace(thickness-0.3, 0, len(vx), False).reshape(-1, 1)
    vz = np.zeros_like(vx)+0.3
    v = np.vstack((v, np.hstack((vx, vy, vz))))
    vy = np.linspace(-thickness+0.3, 0, len(vx), False).reshape(-1, 1)
    vz = np.zeros_like(vx)+0.3
    v = np.vstack((v, np.hstack((vx, vy, vz))))

    vx = np.arange(0, 1-0.8, interval).reshape(-1, 1)
    vy = np.linspace(thickness-0.4, 0, len(vx), False).reshape(-1, 1)
    vz = np.zeros_like(vx)+0.4
    v = np.vstack((v, np.hstack((vx, vy, vz))))
    vy = np.linspace(-thickness+0.4, 0, len(vx), False).reshape(-1, 1)
    vz = np.zeros_like(vx)+0.4
    v = np.vstack((v, np.hstack((vx, vy, vz))))


    vx = np.arange(0, 1-0.2, interval).reshape(-1, 1)
    vy = np.linspace(thickness-0.1, 0, len(vx), False).reshape(-1, 1)
    vz = np.zeros_like(vx)-0.1
    v = np.vstack((v, np.hstack((vx, vy, vz))))
    vy = np.linspace(-thickness+0.1, 0, len(vx), False).reshape(-1, 1)
    vz = np.zeros_like(vx)-0.1
    v = np.vstack((v, np.hstack((vx, vy, vz))))

    vx = np.arange(0, 1-0.4, interval).reshape(-1, 1)
    vy = np.linspace(thickness-0.2, 0, len(vx), False).reshape(-1, 1)
    vz = np.zeros_like(vx)-0.2
    v = np.vstack((v, np.hstack((vx, vy, vz))))
    vy = np.linspace(-thickness+0.2, 0, len(vx), False).reshape(-1, 1)
    vz = np.zeros_like(vx)-0.2
    v = np.vstack((v, np.hstack((vx, vy, vz))))

    vx = np.arange(0, 1-0.6, interval).reshape(-1, 1)
    vy = np.linspace(thickness-0.3, 0, len(vx), False).reshape(-1, 1)
    vz = np.zeros_like(vx)-0.3
    v = np.vstack((v, np.hstack((vx, vy, vz))))
    vy = np.linspace(-thickness+0.3, 0, len(vx), False).reshape(-1, 1)
    vz = np.zeros_like(vx)-0.3
    v = np.vstack((v, np.hstack((vx, vy, vz))))

    vx = np.arange(0, 1-0.8, interval).reshape(-1, 1)
    vy = np.linspace(thickness-0.4, 0, len(vx), False).reshape(-1, 1)
    vz = np.zeros_like(vx)-0.4
    v = np.vstack((v, np.hstack((vx, vy, vz))))
    vy = np.linspace(-thickness+0.4, 0, len(vx), False).reshape(-1, 1)
    vz = np.zeros_like(vx)-0.4
    v = np.vstack((v, np.hstack((vx, vy, vz))))

    for l, G in zip(links[::-1], Gs[::-1]):
        vertices_segments += [k] * len(v)
        k -= 1
        # create vertices
        T = np.vstack((T, v))
        T = trans(T, G)

        v = np.empty((0, 3))
        vx = np.arange(0, l[0] + interval, interval).reshape(-1, 1)
        vy = np.full_like(vx, thickness)
        vz = np.zeros_like(vx)
        v = np.vstack((v, np.hstack((vx, vy, vz))))
        vy = np.full_like(vx, -thickness)
        vz = np.zeros_like(vy)
        v = np.vstack((v, np.hstack((vx, vy, vz))))

        vx = np.arange(0, l[0] + interval, interval).reshape(-1, 1)
        vy = np.full_like(vx, thickness-0.1)
        vz = np.zeros_like(vx)+0.1
        v = np.vstack((v, np.hstack((vx, vy, vz))))
        vy = np.full_like(vx, -thickness+0.1)
        vz = np.zeros_like(vy)+0.1
        v = np.vstack((v, np.hstack((vx, vy, vz))))

        vx = np.arange(0, l[0] + interval, interval).reshape(-1, 1)
        vy = np.full_like(vx, thickness-0.2)
        vz = np.zeros_like(vx)+0.2
        v = np.vstack((v, np.hstack((vx, vy, vz))))
        vy = np.full_like(vx, -thickness+0.2)
        vz = np.zeros_like(vy)+0.2
        v = np.vstack((v, np.hstack((vx, vy, vz))))

        vx = np.arange(0, l[0] + interval, interval).reshape(-1, 1)
        vy = np.full_like(vx, thickness-0.3)
        vz = np.zeros_like(vx)+0.3
        v = np.vstack((v, np.hstack((vx, vy, vz))))
        vy = np.full_like(vx, -thickness+0.3)
        vz = np.zeros_like(vy)+0.3
        v = np.vstack((v, np.hstack((vx, vy, vz))))

        vx = np.arange(0, l[0] + interval, interval).reshape(-1, 1)
        vy = np.full_like(vx, thickness-0.4)
        vz = np.zeros_like(vx)+0.4
        v = np.vstack((v, np.hstack((vx, vy, vz))))
        vy = np.full_like(vx, -thickness+0.4)
        vz = np.zeros_like(vy)+0.4
        v = np.vstack((v, np.hstack((vx, vy, vz))))


        vx = np.arange(0, l[0] + interval, interval).reshape(-1, 1)
        vy = np.full_like(vx, thickness-0.1)
        vz = np.zeros_like(vx)-0.1
        v = np.vstack((v, np.hstack((vx, vy, vz))))
        vy = np.full_like(vx, -thickness+0.1)
        vz = np.zeros_like(vy)-0.1
        v = np.vstack((v, np.hstack((vx, vy, vz))))

        vx = np.arange(0, l[0] + interval, interval).reshape(-1, 1)
        vy = np.full_like(vx, thickness-0.2)
        vz = np.zeros_like(vx)-0.2
        v = np.vstack((v, np.hstack((vx, vy, vz))))
        vy = np.full_like(vx, -thickness+0.2)
        vz = np.zeros_like(vy)-0.2
        v = np.vstack((v, np.hstack((vx, vy, vz))))

        vx = np.arange(0, l[0] + interval, interval).reshape(-1, 1)
        vy = np.full_like(vx, thickness-0.3)
        vz = np.zeros_like(vx)-0.3
        v = np.vstack((v, np.hstack((vx, vy, vz))))
        vy = np.full_like(vx, -thickness+0.3)
        vz = np.zeros_like(vy)-0.3
        v = np.vstack((v, np.hstack((vx, vy, vz))))

        vx = np.arange(0, l[0] + interval, interval).reshape(-1, 1)
        vy = np.full_like(vx, thickness-0.4)
        vz = np.zeros_like(vx)-0.4
        v = np.vstack((v, np.hstack((vx, vy, vz))))
        vy = np.full_like(vx, -thickness+0.4)
        vz = np.zeros_like(vy)-0.4
        v = np.vstack((v, np.hstack((vx, vy, vz))))

    N = len(vertices_segments)
    W = np.zeros((N, K), np.float32)
    W[range(N), vertices_segments] = 1.0

    return J, T, angles, W


if __name__ == '__main__':
    omega_desires = np.array([[0, 0, 0],  # root
                              [0, -1, 1],  # 1
                              [0, 0, 0],  # 2
                              [0, 0, 0]])  # 3)
    temp_theta = np.zeros_like(omega_desires)
    J, T_bar, omega_stars, W = create_joints_and_vertices(False)

    Ts = []
    G = np.eye(4)
    G_star_inv = np.eye(4)
    G_des = np.eye(4)
    for omega_des, omega_temp, j in zip(omega_desires, temp_theta, J):
        print(j)
#        omega_des = np.deg2rad(omega_des)
        j = trans(j, G_star_inv)
        print(j)
        print()
        G_star_k = transform(j[0], omega_temp)
        G_des_k = transform(j[0], omega_des)
        G_star_inv = np.linalg.inv(G_star_k).dot(G_star_inv)
        G_des = G_des.dot(G_des_k)
        G = G_des.dot(G_star_inv)
        T_k = trans(T_bar, G)
        Ts.append(T_k)

    for t in Ts:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot(*J.T, 'o-')
        ax.plot(*t.T, '.')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(-3, 3)
        plt.gca().set_aspect(1)
        plt.grid()
        plt.show()

    # use einsum instead of for loop
    T_prime = np.zeros_like(T_bar)
    for t, w in zip(Ts, W.T):

        T_prime += t * w.reshape(-1, 1)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_title('T_bar')
    ax.plot(*J.T, 'o-')
    ax.plot(*T_bar.T, '.')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    plt.gca().set_aspect(1)
    plt.grid()
    plt.show()

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_title('T_prime')
    ax.plot(*J.T, 'o-')
    ax.plot(*T_prime.T, '.')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    plt.gca().set_aspect(1)
    plt.grid()
    plt.show()
