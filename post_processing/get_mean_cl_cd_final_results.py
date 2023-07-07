"""
    compute the mean & std. values for cl & cd for the simulations using the final policy
"""
import pandas as pd

from glob import glob
from os.path import join
from typing import Union


def compute_mean_std_coefficients(cases: list, env: str = "rotatingCylinder2D", t_start: Union[int, float] = 8) -> None:
    """
    compute the mean cl & cd values and the corresponding std. deviation of the simulations using the final policy

    :param cases: list containing the top-level directories of the cases to evaluate
    :param env: either 'rotatingCylinder2D' or 'rotatingPinball2D'
    :param t_start: time for start computing the avg. & std. deviation
    :return: None
    """
    controlled = []
    for c in cases:
        # import the trajectories of the controlled cases
        if env == "rotatingCylinder2D":
            if c == "uncontrolled":
                load_path = join("..", "data", env, c)
            else:
                load_path = glob(join("..", "data", env, c, "*", "results_best_policy"))[0]
        else:
            if c == "uncontrolled":
                load_path = join("..", "data", env, c)
            else:
                load_path = join("..", "data", env, c, "results_final_policy")
        controlled.append(pd.read_csv(join(load_path, "postProcessing", "forces", "0", "coefficient.dat"), skiprows=13,
                                      header=0, sep=r"\s+", usecols=[0, 1, 2], names=["t", "cd", "cl"]))

    # get the mean & std cl- and cd for t > x s
    idx = [controlled[c][controlled[c]["t"] == t_start].index.values.item() for c in range(len(controlled))]

    # print results
    print(55 * "=")
    print(f"\t\tresults for environment {env}")
    print(55 * "=")
    for i in range(len(idx)):
        print(f"\ncase {i}:\n-------")
        print(f"\tcl_mean = {round(controlled[i]['cl'][idx[i]:].mean(), 4)}, "
              f"cl_std = {round(controlled[i]['cl'][idx[i]:].std(), 4)}")
        print(f"\tcd_mean = {round(controlled[i]['cd'][idx[i]:].mean(), 4)}, "
              f"cd_std = {round(controlled[i]['cd'][idx[i]:].std(), 4)}")
    print("\n")


if __name__ == "__main__":
    cases_cylinder = ["uncontrolled", "e200_r10_b10_f8_MF", "e200_r10_b10_f8_MB_1model",
                      "e200_r10_b10_f8_MB_5models_thr3", "e200_r10_b10_f8_MB_5models_thr2",
                      "e200_r10_b10_f8_MB_10models_thr6", "e200_r10_b10_f8_MB_10models_thr5",
                      "e200_r10_b10_f8_MB_10models_thr3"]

    cases_pinball = ["uncontrolled", "e150_r10_b10_f300_MF", "e150_r10_b10_f300_MB_1model",
                     "e150_r10_b10_f300_MB_5models_thr2", "e150_r10_b10_f300_MB_10models_thr5"]

    # load the cl & cd values for the results using the final policies
    compute_mean_std_coefficients(cases_cylinder)
    compute_mean_std_coefficients(cases_pinball, env="rotatingPinball2D", t_start=350)

