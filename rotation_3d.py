# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 18:29:22 2018

@author: ryuhei
"""

import numpy as np
import matplotlib.pyplot as plt

#import chainer
#import chainer.functions as F
#import chainer.links as L
#from chainer import cuda, Variable


def cross_product_matrix(v_unit):
    assert np.allclose(np.linalg.norm(v_unit), 1.0), np.linalg.norm(v_unit)
    return np.array([[0, -v_unit[2], v_unit[1]],
                     [v_unit[2], 0, -v_unit[0]],
                     [-v_unit[1], v_unit[0], 0]], v_unit.dtype)


def axis_angle_to_rotation_matrix(omega):
    """Compute a rotation matrix from an axis-angle vector by Rodrigues formula
    """

    theta = np.linalg.norm(omega)
    if theta == 0:
        return np.eye(3, dtype=omega.dtype)

    omega_unit = omega / theta
    K = cross_product_matrix(omega_unit)
    I = np.eye(3, dtype=omega.dtype)
    R = I + np.sin(theta) * K + (1 - np.cos(theta)) * K.dot(K)
    return R


def rotate(x, rotation_matrix):
    return x.dot(rotation_matrix.T)


if __name__ == '__main__':
    # "Right-hand rule" is assumed.
    # Right hand coordinate system (i.e. x-axis corresponds to right thumb).
    # Positive angle direction of rotation: When right thumb points along
    # a rotation axis, then the curl direction that other fingers are pointing
    # is the positive direction of angle.
    # A vertex is represented as a row vector.

    x = np.array([[1, -1, 0],
                  [1.5, -1.5, 0],
                  [1.3, -1.0, 0],
                  [1.0, -1.3, 0]])
    r_axis = np.array([1, 0.5, 0], 'f')  # unnormalized vector is ok
    r_angle = 10  # in degree

    for r_angle in np.arange(0, 360, 10):
        omega_unit = r_axis / np.linalg.norm(r_axis)
        omega = omega_unit * np.deg2rad(r_angle)
        print('omega', omega)
        R = axis_angle_to_rotation_matrix(omega)
        y = rotate(x, R)

        plt.plot(y[:, 0], y[:, 1], '.-')
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.gca().set_aspect(1)
        plt.grid()
        plt.show()
        print(y)
