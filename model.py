# --- Python 3.8 ---
"""
@File : model.py
@Time : 2021/04/06
@Author : Peter Atma
@Desc : None
"""

# --- Standard Python modules ---
# --- External Python modules ---
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker


# import pint

# --- Extension modules ---

# --- Code ---
rho_s = 1000
rho_p = 800
mL = 10
r = 0.5


def mass_ratio(L1, L2):
    # --- Stage 2 ---
    # L2 = 2.0
    t2 = 0.005

    v2 = np.pi * (r ** 2 * L2 + 4 / 3 * r ** 3)
    vp2 = np.pi * (L2 * (r - t2) ** 2 + 4 / 3 * (r - t2) ** 3)
    vs2 = v2 - vp2

    ms2 = vs2 * rho_s
    mp2 = vp2 * rho_p

    m02 = ms2 + mp2 + mL
    # mb2 = mL + ms2

    # R2 = m02 / mb2
    e2 = ms2 / (ms2 + mp2)

    # --- Stage 1 ---
    # L1 = 8.0
    t1 = 0.01

    v1 = np.pi * (r ** 2 * L1 + 4 / 3 * r ** 3)
    vp1 = np.pi * (L1 * (r - t1) ** 2 + 4 / 3 * (r - t1) ** 3)
    vs1 = v1 - vp1

    ms1 = vs1 * rho_s
    mp1 = vp1 * rho_p

    m01 = ms1 + mp1 + m02
    # mb1 = m02 + ms1

    # R1 = m01 / mb1
    e1 = ms1 / (ms1 + mp1)
    print(e1, e2)
    return m01 / mL


# t_1 = np.linspace(0, 0.01, 100)
# t_2 = np.linspace(0, 0.01, 100)
# T1, T2 = np.meshgrid(t_1, t_2)

l1 = np.linspace(1, 8, 100)
l2 = np.linspace(1, 3, 100)
L1, L2 = np.meshgrid(l1, l2)

# obj = mass_ratio(L1, L2)

# plt.figure(figsize=(8, 6))
# plt.contour(L1, L2, obj, locator=ticker.LogLocator(), levels=100)
# plt.colorbar()
# plt.show()

print(mass_ratio(8, 2))
