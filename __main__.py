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
        ApertureRadius = 20  # mm

        C4 = 0
        C11 = (-0.01395786 * 5**0.5)*0.525e-6
        C22 = 0

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
    R = (X**2 + Y**2) ** 0.5
    Z = (CONSTANTS.C4 * (2 * R**2 - 1) +
         CONSTANTS.C11 * (6*R**4 - 6*R**2 + 1))

    # Aperture realization
    _indexes = np.where(((X + CONSTANTS.ApertureX)**2 + (Y + CONSTANTS.ApertureY)**2)**0.5 > CONSTANTS.ApertureRadius)
    Z[_indexes] = np.nan


    dZy, dZx = np.gradient(Z)
    dY, dX = np.gradient(X)


    Der=-dZx/dX

    Der_mean = Der

    Deviation = np.mean(np.ravel(Der_mean[~np.isnan(Der_mean)]))


    Fi=Deviation*3600

    return Fi


Distance_list = [200]
ApertureDiameter = 20
ApertureDistancetoSUT = 10

for distance in Distance_list:
    for shift in [-8, -4, -2, 0, 2, 4, 8]:
        X = []
        Y = list()
        for i in tqdm(np.linspace(-1300, 1300, 40), desc = f"Analyzing for shift {shift} mm", leave = False):
            Da = ApertureDiameter - np.abs(np.tan(math.radians(2 * i/3600)) * ApertureDistancetoSUT)
            dX = (distance * np.tan(math.radians(2 * i/3600)) +
                  shift - ApertureDiameter/2 +
                  Da/2)
            Y.append(mainAction(fQ = 500,
                                Radius = 20,
                                ApertureRadius = Da/2,
                                ApertureY= 0,
                                ApertureX= dX))
            X.append(i)
        Y = detrend(Y)
        plt.plot(X,Y, label = f"{shift} mm")
    plt.grid()
    plt.legend()
    plt.show()