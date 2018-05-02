# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 16:27:50 2018

@author: yamane
"""

import _pickle as pickle


def temp(dd):
    dd['J_regressor_prior'] = dd[b'J_regressor_prior']
    del dd[b'J_regressor_prior']
    dd['f'] = dd[b'f']
    del dd[b'f']
    dd['J_regressor'] = dd[b'J_regressor']
    del dd[b'J_regressor']
    dd['kintree_table'] = dd[b'kintree_table']
    del dd[b'kintree_table']
    dd['J'] = dd[b'J']
    del dd[b'J']
    dd['weights_prior'] = dd[b'weights_prior']
    del dd[b'weights_prior']
    dd['weights'] = dd[b'weights']
    del dd[b'weights']
    dd['vert_sym_idxs'] = dd[b'vert_sym_idxs']
    del dd[b'vert_sym_idxs']
    dd['posedirs'] = dd[b'posedirs']
    del dd[b'posedirs']
    dd['pose_training_info'] = dd[b'pose_training_info']
    del dd[b'pose_training_info']
    dd['bs_style'] = dd[b'bs_style']
    del dd[b'bs_style']
    dd['v_template'] = dd[b'v_template']
    del dd[b'v_template']
    dd['shapedirs'] = dd[b'shapedirs']
    del dd[b'shapedirs']
    dd['bs_type'] = 'lrotmin'
    del dd[b'bs_type']
    return dd


def J_regressor_prior(a):
    a.__dict__['_shape'] = a.__dict__[b'_shape']
    del a.__dict__[b'_shape']
    a.__dict__['data'] = a.__dict__[b'data']
    del a.__dict__[b'data']
    a.__dict__['format'] = 'csc'
    del a.__dict__[b'format']
    a.__dict__['indices'] = a.__dict__[b'indices']
    del a.__dict__[b'indices']
    a.__dict__['indptr'] = a.__dict__[b'indptr']
    del a.__dict__[b'indptr']
    a.__dict__['maxprint'] = a.__dict__[b'maxprint']
    del a.__dict__[b'maxprint']
    return a


def J_regressor(a):
    a.__dict__['_shape'] = a.__dict__[b'_shape']
    del a.__dict__[b'_shape']
    a.__dict__['data'] = a.__dict__[b'data']
    del a.__dict__[b'data']
    a.__dict__['format'] = 'csc'
    del a.__dict__[b'format']
    a.__dict__['indices'] = a.__dict__[b'indices']
    del a.__dict__[b'indices']
    a.__dict__['indptr'] = a.__dict__[b'indptr']
    del a.__dict__[b'indptr']
    a.__dict__['maxprint'] = a.__dict__[b'maxprint']
    del a.__dict__[b'maxprint']
    return a


def shapedirs(a):
    a.__dict__['x'] = a.__dict__[b'x']
    del a.__dict__[b'x']
    a.__dict__['_dirty_vars'] = 'x'
    del a.__dict__[b'_dirty_vars']
    a.__dict__['_itr'] = a.__dict__[b'_itr']
    del a.__dict__[b'_itr']
    a.__dict__['_depends_on_deps'] = a.__dict__[b'_depends_on_deps']
    del a.__dict__[b'_depends_on_deps']
    return a

def face(a):
    a.__dict__['f'] = a.__dict__[b'f']
    del a.__dict__[b'f']
    a.__dict__['_dirty_vars'] = 'x'
    del a.__dict__[b'_dirty_vars']
    a.__dict__['_itr'] = a.__dict__[b'_itr']
    del a.__dict__[b'_itr']
    a.__dict__['_depends_on_deps'] = a.__dict__[b'_depends_on_deps']
    del a.__dict__[b'_depends_on_deps']
    return a

def load(fname_or_dict):
    dd = pickle.load(open(fname_or_dict, 'rb'), encoding='bytes')
    dd = temp(dd)
    dd['J_regressor_prior'] = J_regressor_prior(dd['J_regressor_prior'])
    dd['J_regressor'] = J_regressor(dd['J_regressor'])
    dd['shapedirs'] = shapedirs(dd['shapedirs'])
    dd = fix_params(dd)
    return dd

def fix_params(m):
    del m['J_regressor_prior']
    del m['J']
    del m['weights_prior']
    del m['vert_sym_idxs']
    del m['pose_training_info']
    del m['bs_style']
    del m['bs_type']

    m['J'] = m['J_regressor']
    del m['J_regressor']
    m['W'] = m['weights']
    del m['weights']
    m['P'] = m['posedirs']
    del m['posedirs']
    m['T_bar'] = m['v_template']
    del m['v_template']
    m['S'] = m['shapedirs']
    del m['shapedirs']

    return m