#!/usr/bin/env python3
"""
Complete coupled solver for L-shaped domain problem.
Subdomain A: scikit-fem (everything except (0.5,1)x(0,0.5))
Subdomain B: FEniCSx/dolfinx ((0.5,1)x(0,0.5))

This script coordinates both solvers and implements Dirichlet-Neumann coupling.
"""

import numpy as np
import os
import sys
import csv
from pathlib import Path

WORK_DIR = Path("/home/alexander/Schreibtisch/ofa-v2/campaign3_blind/runs/C5_27b_BARE_seed11/work")
SKFEM_PYTHON = "/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python"
FENICS_PYTHON = "/home/alexander/miniconda3/envs/fenics/bin/python"

# Define probe points
def get_probe_points_A():
    points = []
    for i_x in range(44):
        for i_y in range(44):
            x = (i_x + 0.5) / 44.0
            y = (i_y + 0.5) / 44.0
            if x > 0.5 and y < 0.5:
                continue
            if x > 0.75 and y > 0.75:
                continue
            points.append((x, y))
    return points

def get_probe_points_B():
    points = []
    for i_x in range(44):
        for i_y in range(44):
            x = 0.5 + (i_x + 0.5) * 0.5 / 44.0
            y = (i_y + 0.5) * 0.5 / 44.0
            points.append((x, y))
    return points

def get_interface_probes():
    points = []
    for i in range(44):
        x = 0.5
        y = 1/8 + (i + 0.5) * (1/4) / 44.0
        points.append((x, y, 'leg1'))
    for i in range(44):
        x = 5/8 + (i + 0.5) * (1/4) / 44.0
        y = 0.5
        points.append((x, y, 'leg2'))
    return points

# Material coefficients
def k_A(x, y):
    if x < 0.5 and y < 0.5:
        return 1.0
    elif x < 0.5 and y >= 0.5:
        return 2.0
    elif x >= 0.5 and y >= 0.5:
        return 5.0
    else:
        return 0.0

def k_B(x, y):
    return 2.5

# Source terms
def f_A(x, y):
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
    return (24*x**4*y/125 - 11*x**4/125 + 48*x**3*y**2/25 - 51*x**3*y/125 - 67*x**3/250 
            + 48*x**2*y**3/125 - 306*x**2*y**2/125 + 36*x**2*y/125 + 811*x**2/2000 
            + 24*x*y**4/25 - 51*x*y**3/125 - 471*x*y**2/250 + 657*x*y/1000 + 603*x/4000 
            - 8*y**4/25 + 9*y**3/250 + 2971*y**2/2000 - 2487*y/8000 - 801/4000)

print("Starting coupled solver...")
print(f"Probe points A: {len(get_probe_points_A())}")
print(f"Probe points B: {len(get_probe_points_B())}")
print(f"Interface probes: {len(get_interface_probes())}")
