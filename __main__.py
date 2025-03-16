import os
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import detrend
from tqdm import tqdm

from matplotlib import cm


def mainAction(fQ,
               Radius: float,
               ApertureX: float,
               ApertureY: float ,
               ApertureRadius: float):

    class CONSTANTS:
        fQ = 1600
        Radius = 20  # mm
        ApertureX = 0  # mm
        ApertureY = 0  # mm
        ApertureRadius = 20 # mm

        C4 = 0
        C11 = (-0.027 * (5 **0.5))*1e-3
        C22 = 0
        C37 = 0

    CONSTANTS.ApertureX = ApertureX
    CONSTANTS.fQ = fQ
    CONSTANTS.Radius = Radius
    CONSTANTS.ApertureY = ApertureY
    CONSTANTS.ApertureRadius = ApertureRadius

    # X, Y series
    X = np.linspace(-CONSTANTS.Radius, CONSTANTS.Radius, CONSTANTS.fQ)
    Y = np.linspace(-CONSTANTS.Radius, CONSTANTS.Radius, CONSTANTS.fQ)

    # Mesh realization
    X,Y = np.meshgrid(X,Y)
    R = (X**2 + Y**2) ** 0.5 / CONSTANTS.Radius
    Z = (CONSTANTS.C4 * (2 * R**2 - 1) +
         CONSTANTS.C11 * (6*R**4 - 6*R**2 + 1) +
         CONSTANTS.C22 * (20 * (R**6) - 30 * (R**4) + 12 * (R**2) - 1))


    # Aperture realization
    _indexes = np.where(((X + CONSTANTS.ApertureX)**2 + (Y + CONSTANTS.ApertureY)**2)**0.5 > CONSTANTS.ApertureRadius)
    Z[_indexes] = np.nan
    X[_indexes] = np.nan

    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # ax.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)
    # plt.show()

    dZx = np.diff(Z)
    dX = np.diff(X)

    Der = -np.arctan(dZx/dX)

    Der_mean = Der

    Deviation = np.degrees(np.mean(np.ravel(Der_mean[~np.isnan(Der_mean)])))


    Fi=Deviation*3600

    return Fi


Distance_list = [200,500, 1000, 1500, 1900]
Shift_list = [0]
ApertureDiameter = 40
ApertureDistancetoSUT = 0

# for distance in Distance_list:
#     for shift in Shift_list:
#         X = []
#         Y = list()
#         for i in tqdm(np.linspace(-1300, 1300, 40), desc = f"Analyzing for shift {shift} mm Distance {distance}", leave = False):
#             Da = ApertureDiameter
#             dX = (distance * np.tan(math.radians(2 * i/3600)))
#             Y.append(mainAction(fQ = 250,
#                                 Radius = 20,
#                                 ApertureRadius = Da/2,
#                                 ApertureY= 0,
#                                 ApertureX= dX))
#             X.append(i)
#
#         X = np.array(X)
#         Y = np.array(Y)
#
#         m,b = np.polyfit(X,Y,1)
#         Y_M = m*X + b
#         plt.plot(X,Y - Y_M, label = f"Shift {shift}  mm Distance {distance} mm")

for distance in Distance_list:
    X = list()
    Y = list()
    for j in tqdm(np.linspace(-0.1, 0.1, 20), f"Distance {distance} mm"):
        sub_X = list()
        sub_Y = list()
        for i in (np.linspace(-1300, 1300, 40)) :
            dX = distance * np.tan(math.radians(2 * i/3600) + np.arctan(j/250))
            sub_Y.append(mainAction(fQ = 250, Radius = 20,
                                ApertureRadius= ApertureDiameter/2,
                                ApertureY = 0,
                                ApertureX = dX))
            sub_X.append(i)
        Y.append(sub_Y)
        X.append(sub_X)

    R = np.zeros(len(Y[0]))
    G = np.zeros(len(X[0]))
    for index, i in enumerate(Y):
        print(i)
        R += np.array(i)
        G += np.array(X[index])
    R = R/len(Y)
    G = G/len(Y)

    m, b = np.polyfit(G, R, 1)
    Y_M = m * G + b
    print(R, G)
    plt.plot(G, R - Y_M, label=f"mm Distance {distance} mm")
plt.grid()
plt.legend()
plt.show()