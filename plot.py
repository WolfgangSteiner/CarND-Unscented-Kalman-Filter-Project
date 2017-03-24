import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import namedtuple

UKFData = namedtuple('UKFData', "pred_x,pred_y,meas_x,meas_y,ground_truth_x,ground_truth_y,nis_lidar,nis_radar")

def read_ukf_data(meas_number):
    output = pd.read_table("output-%d.txt" % meas_number)
    pred_x = output['pred_x']
    pred_y = output['pred_y']
    meas_x = output['meas_x']
    meas_y = output['meas_y']
    ground_truth_x = output['ground_truth_x']
    ground_truth_y = output['ground_truth_y']
    nis_radar = pd.read_table("nis_radar-%d.txt" % meas_number)['NIS_Radar']
    nis_lidar = pd.read_table("nis_lidar-%d.txt" % meas_number)['NIS_Lidar']

    return UKFData(
        pred_x, pred_y,
        meas_x, meas_y,
        ground_truth_x, ground_truth_y,
        nis_radar, nis_lidar)


def plot_ukf_data(meas_number, ax):
    ukf_data = read_ukf_data(meas_number)
    col = meas_number - 1
    ax[0,col].set_title("Dataset %d" % meas_number)
    ax[0,col].plot(ukf_data.ground_truth_x, ukf_data.ground_truth_y, label="ground truth")
    ax[0,col].plot(ukf_data.meas_x, ukf_data.meas_y, ".", label="measurement")
    ax[0,col].plot(ukf_data.pred_x, ukf_data.pred_y, label="prediction")
    ax[0,col].legend(loc="lower right")

    chi_square_radar = 7.815
    x1, x2 = 0, len(ukf_data.nis_radar) - 1
    y = chi_square_radar
    ax[1,col].plot(ukf_data.nis_radar, label="NIS Radar")
    ax[1,col].plot([x1,x2],[y,y], label="chi^2 0.05")
    ax[1,col].legend()
    ax[1,col].set_ylim([0,10])

    chi_square_lidar = 5.991
    x1, x2 = 0, len(ukf_data.nis_lidar) - 1
    y = chi_square_lidar
    ax[2,col].plot(ukf_data.nis_lidar, label="NIS Lidar")
    ax[2,col].plot([x1,x2],[y,y], label="chi^2 0.05")
    ax[2,col].legend()
    ax[2,col].set_ylim([0,10])


fig,ax = plt.subplots(3,2)
fig.set_size_inches(12, 12)
plot_ukf_data(1,ax)
plot_ukf_data(2,ax)

plt.savefig("plot.png")
