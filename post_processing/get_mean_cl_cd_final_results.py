"""
    compute the mean & std. values for cl & cd for the simulations using the final policy (rotatingCylinder2D)
"""
import pandas as pd

from glob import glob
from os.path import join


if __name__ == "__main__":
    cases = ["e200_r10_b10_f8_MF", "e200_r10_b10_f8_MB_1model", "e200_r10_b10_f8_MB_5models_thr3",
             "e200_r10_b10_f8_MB_5models_thr2", "e200_r10_b10_f8_MB_10models_thr6", "e200_r10_b10_f8_MB_10models_thr5",
             "e200_r10_b10_f8_MB_10models_thr3"]

    controlled = []
    for c in cases:
        # import the trajectories of the controlled cases
        load_path = glob(join("..", "data", "rotatingCylinder2D", c, "*", "results_best_policy"))[0]
        controlled.append(pd.read_csv(join(load_path, "postProcessing", "forces", "0", "coefficient.dat"), skiprows=13,
                                      header=0, sep=r"\s+", usecols=[0, 1, 2], names=["t", "cd", "cl"]))

    # get the mean & std cl- and cd for t > 8s
    idx = [controlled[c][controlled[c]["t"] == 8.0].index.values.item() for c in range(len(controlled))]

    # print results
    for i in range(len(idx)):
        print(f"\ncase {i}:\n-------")
        print(f"\tcl_mean = {round(controlled[i]['cl'][idx[i]:].mean(), 4)}, "
              f"cl_std = {round(controlled[i]['cl'][idx[i]:].std(), 4)}")
        print(f"\tcd_mean = {round(controlled[i]['cd'][idx[i]:].mean(), 4)}, "
              f"cd_std = {round(controlled[i]['cd'][idx[i]:].std(), 4)}")
