#!/usr/bin/env python3
"""
Main coupled solver script.
This script coordinates between scikit-fem (subdomain A) and FEniCSx (subdomain B).
"""

import numpy as np
import os
import sys
import subprocess
import pickle
import csv

# Define probe points
def get_probe_points_A():
    """Get probe points for subdomain A (1331 points after exclusions)"""
    points = []
    for i_x in range(44):
        for i_y in range(44):
            x = (i_x + 0.5) / 44.0
            y = (i_y + 0.5) / 44.0
            # Exclude points in subdomain B: (0.5, 1) x (0, 0.5)
            if x > 0.5 and y < 0.5:
                continue
            # Exclude points in removed square: (0.75, 1) x (0.75, 1)
            if x > 0.75 and y > 0.75:
                continue
            points.append((x, y))
    return points

def get_probe_points_B():
    """Get probe points for subdomain B (1936 points)"""
    points = []
    for i_x in range(44):
        for i_y in range(44):
            x = 0.5 + (i_x + 0.5) * 0.5 / 44.0
            y = (i_y + 0.5) * 0.5 / 44.0
            points.append((x, y))
    return points

def get_interface_probes():
    """Get interface probe points (88 points total)"""
    points = []
    # Leg 1: x = 1/2, y from 1/8 to 3/8 (interior of vertical leg)
    for i in range(44):
        x = 0.5
        y = 1/8 + (i + 0.5) * (1/4) / 44.0
        points.append((x, y, 'leg1'))
    # Leg 2: y = 1/2, x from 5/8 to 7/8 (interior of horizontal leg)
    for i in range(44):
        x = 5/8 + (i + 0.5) * (1/4) / 44.0
        y = 0.5
        points.append((x, y, 'leg2'))
    return points

# Material coefficients
def k_A(x, y):
    """Conductivity on subdomain A"""
    if x < 0.5 and y < 0.5:
        return 1.0
    elif x < 0.5 and y >= 0.5:
        return 2.0
    elif x >= 0.5 and y >= 0.5:
        return 5.0
    else:
        return 0.0

def k_B(x, y):
    """Conductivity on subdomain B - always 5/2"""
    return 2.5

# Source terms
def f_A(x, y):
    """Source term on subdomain A"""
    if x < 0.5 and y < 0.5:
        return (3*x**4*y - 11*x**4/8 + 12*x**3*y**2 - 123*x**3*y/20 - x**3/40 
                + 6*x**2*y**3 - 477*x**2*y**2/20 + 2799*x**2*y/400 + 1423*x**2/800 
                + 6*x*y**4 - 123*x*y**3/20 + 993*x*y**2/200 + 63*x*y/1600 - 609*x/800 
                - 13*y**4/5 + 279*y**3/200 + 1423*y**2/800 - 327*y/320)
    elif x < 0.5 and y >= 0.5:
        return (3*x**4*y/4 - 5*x**4/16 + 3*x**3*y**2/2 - 3*x**3*y/80 - 13*x**3/32 
                + 3*x**2*y**3/2 - 153*x**2*y**2/40 - 873*x**2*y/800 + 119*x**2/80 
                + 3*x*y**4/4 - 3*x*y**3/80 - 471*x*y**2/800 + 9*x*y/16 - 3*x/800 
                - 13*y**4/40 - 241*y**3/800 + 37*y**2/40 - 107*y/3200 - 849/3200)
    elif x >= 0.5 and y >= 0.5:
        return (6*x**4*y/125 - x**4/50 + 6*x**3*y**2/25 + 69*x**3*y/500 - x**3/8 
                + 12*x**2*y**3/125 - 9*x**2*y**2/25 - 9*x**2*y/40 + 769*x**2/4000 
                + 3*x*y**4/25 + 69*x*y**3/500 - 51*x*y**2/100 - 549*x*y/8000 + 2853*x/16000 
                - y**4/25 - 71*y**3/1000 + 233*y**2/800 + 419*y/4000 - 2031/16000)
    else:
        return 0.0

def f_B(x, y):
    """Source term on subdomain B"""
    return (24*x**4*y/125 - 11*x**4/125 + 48*x**3*y**2/25 - 51*x**3*y/125 - 67*x**3/250 
            + 48*x**2*y**3/125 - 306*x**2*y**2/125 + 36*x**2*y/125 + 811*x**2/2000 
            + 24*x*y**4/25 - 51*x*y**3/125 - 471*x*y**2/250 + 657*x*y/1000 + 603*x/4000 
            - 8*y**4/25 + 9*y**3/250 + 2971*y**2/2000 - 2487*y/8000 - 801/4000)

print("Probe points A:", len(get_probe_points_A()))
print("Probe points B:", len(get_probe_points_B()))
print("Interface probes:", len(get_interface_probes()))
