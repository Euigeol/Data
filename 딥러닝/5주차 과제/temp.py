# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
def AND(x1, x2):  # AND 
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    temp = np.sum(w*x) + b
    if temp <= 0:
        return 0
    else:
        return 1

def NAND(x1, x2): # NAND
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    temp = np.sum(w*x) + b
    if temp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):  # OR 
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    temp = np.sum(w*x) + b
    if temp <= 0:
        return 0
    else:
        return 1

def XOR(x1, x2): # XOR 
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

def Full_Adder(X, Y, Z):
    SUM = XOR(XOR(X, Y), Z)
    CARRY = OR(AND(X, Y), AND(XOR(X, Y), Z))
    return CARRY, SUM

# 결과 출력
print('X = 0, Y = 0, Z = 0  => ')
print((Full_Adder(0, 0, 0)))

print('\nX = 0, Y = 1, Z = 0  => ')
print((Full_Adder(0, 1, 0)))

print('\nX = 1, Y = 0, Z = 0  => ')
print((Full_Adder(1, 0, 0)))

print('\nX = 1, Y = 0, Z = 1  => ')
print((Full_Adder(1, 0, 1)))

print('\nX = 1, Y = 1, Z = 0  => ')
print((Full_Adder(1, 1, 0)))

print('\nX = 0, Y = 1, Z = 1  => ')
print((Full_Adder(0, 1, 1)))

print('\nX = 1, Y = 1, Z = 1  => ')
print((Full_Adder(1, 1, 1)))

