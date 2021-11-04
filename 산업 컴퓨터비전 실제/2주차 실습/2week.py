# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 20:37:33 2021

@author: ATMS
"""

import argparse

import cv2

parser = argparse.ArgumentParser()

parser.add_argument('--path', default='lena.png', help='Image path.')
params = parser.parse_args()
 
img = cv2.imread(params.path)
 
assert img is not None
print('read {}'.format(params.path))
print('shape:', img.shape)
print('drype:', img.dtype)
 
img = cv2.imread(params.path, cv2.IMREAD_GRAYSCALE)
assert img is not None
 
print('read {} as grayscale'.format(params.path))
print('shape:', img.shape)
print('dtype:', img.dtype)
