# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 16:25:50 2018

@author: yamane
"""

import numpy as np
from utils import load
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def trans(x, mat):
    x = np.atleast_2d(x)
    n = len(x)
    x_hom = np.hstack((x, np.ones((n, 1))))
    return x_hom.dot(mat.T)[:, :-1]

def skew(p):
    #  3 次元ベクトルを歪対称行列に変換
    V = np.array([[0, -p[2], p[1]],
                  [p[2], 0, -p[0]],
                  [-p[1], p[0], 0]])
    return V

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

def shape_blend_shapes(b, s):
    Bs = 0
    s = np.transpose(s, (2, 0, 1))
    for i in range(len(b)):
        Bs +=  b[i] * s[i]
    return Bs

def joint_locations(b, j, t_bar, s):
    Bs = shape_blend_shapes(b, s)
    t = t_bar + Bs
    J = j.dot(t)
    return J

def pose_blend_shapes(theta, p):
    Bp = 0
    temp_theta = np.zeros_like(theta)
    rs = []
    rs_star = []
    p = np.transpose(p, (2, 0, 1))
    for omega in theta[1:]:
        r = rotation(omega)
        rs.append(r)
    rs = np.array(rs)
    R = np.ndarray.flatten(rs)

    for omega in temp_theta[1:]:
        r = rotation(omega)
        rs_star.append(r)
    rs_star = np.array(rs_star)
    R_star = np.ndarray.flatten(rs_star)

    for i in range(len(p)):
        Bp += (R[i] - R_star[i]) * p[i]
    return Bp

def blend_skinning(T_bar, J, omega_desires, W):
    Ts = []
    temp_theta = np.zeros_like(theta)
    G = np.eye(4)
    G_star_inv = np.eye(4)
    G_des = np.eye(4)
    for omega_des, omega_temp, j in zip(omega_desires, temp_theta, J):
        j = trans(j, G_star_inv)
        G_star_k = transform(j[0], omega_temp)
        G_des_k = transform(j[0], omega_des)
        G_star_inv = np.linalg.inv(G_star_k).dot(G_star_inv)
        G_des = G_des.dot(G_des_k)
        G = G_des.dot(G_star_inv)
        T_k = trans(T_bar, G)
        Ts.append(T_k)

#    for t in Ts:
#        fig = plt.figure()
#        ax = Axes3D(fig)
#        ax.plot(*J.T, 'o-')
#        ax.plot(*t.T, '.')
#        ax.set_xlim(-1, 1)
#        ax.set_ylim(-1, 1)
#        ax.set_zlim(-1, 1)
#        plt.gca().set_aspect(1)
#        plt.grid()
#        plt.show()

    # use einsum instead of for loop
    T_prime = np.zeros_like(T_bar)
    for t, w in zip(Ts, W.T):
        T_prime += t * w.reshape(-1, 1)

    return T_prime


if __name__ == '__main__':
    fname_or_dict = 'basicModel_m_lbs_10_207_0_v1.0.0.pkl'
    beta_size = 10
    pose_para = np.array([[0., 0, 0],  # root
                          [0, 0, 0],  # 1
                          [0, 0, 0],  # 2
                          [1, 0, 0],  # 3
                          [0, 0, 0],  # 4
                          [0, 0, 0],  # 5
                          [0, 0, 0],  # 6
                          [0, 0, 0],  # 7
                          [0, 0, 0],  # 8
                          [0, 0, 0],  # 9
                          [0, 0, 0],  # 10
                          [0, 0, 0],  # 11
                          [0, 0, 0],  # 12
                          [0, 0, 0],  # 13
                          [0, 0, 0],  # 14
                          [0, 0, 0],  # 15
                          [0, 0, 0],  # 16
                          [0, 0, 0],  # 17
                          [0, 0, 0],  # 18
                          [0, 0, 0],  # 19
                          [0, 0, 0],  # 20
                          [0, 0, 0],  # 21
                          [0, 0, 0],  # 22
                          [0, 0, 0]])  # 23

    m = load(fname_or_dict)
    J = m['J']  # shape=(24, 6890)
    P = m['P']  # shape=(6890, 3, 207)
    S = np.array(m['S'])  # shape=(6890, 3, 10)
    T_bar = m['T_bar']  # shape=(6890, 3)
    W = m['W']  # shape=(6890, 24)
    Kintree_table = m['kintree_table']  # shape=(2, 24)

    ## Assign random pose and shape parameters
    theta = np.array(pose_para)  # 関節角
    beta = np.random.rand(beta_size) * .1  # 体形パラメータ

    Bs = shape_blend_shapes(beta, S)  # shape=(6890, 3)
    Joint = joint_locations(beta, J, T_bar, S)  # shape=(24, 3)
    Bp = pose_blend_shapes(theta, P)  # shape=(6890, 3)
    Tp = T_bar + Bs + Bp  # shape=(6890, 3)
    T = blend_skinning(Tp, Joint, theta, W)  # shape=(6890, 3)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_title('T_prime')
    ax.plot(*T.T, '.')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    plt.gca().set_aspect(1)
    plt.grid()
    plt.show()
