"""
    get the mean & std. values for cl & cd from the cylinder2D case
"""
import pandas as pd


if __name__ == "__main__":
    setup = {
        "main_load_path": r"/home/janis/Hiwi_ISM/results_drlfoam_MB/",
        "path_controlled": r"run/final_routine_AWS/",       # path to top-level dir containing all the cases
        "path_final_results": r"results_best_policy/",  # path to the dir of each case using the best policy
        "case_name": ["e200_r10_b10_f8_MB_5models_threshold40/seed1/"], # cases
        "n_probes": 12,  # number of probes placed in flow field
    }

    controlled = []
    for case in range(len(setup["case_name"])):
        # import the trajectories of the controlled cases
        controlled.append(pd.read_csv("".join([setup["main_load_path"], setup["path_controlled"],
                                               setup["case_name"][case], setup["path_final_results"],
                                               r"postProcessing/forces/0/coefficient.dat"]), skiprows=13, header=0,
                                      sep=r"\s+", usecols=[0, 1, 2], names=["t", "cd", "cl"]))

    # get the mean & std cl- and cd for t > 8s
    idx = [controlled[c][controlled[c]["t"] == 8.0].index.values.item() for c in range(len(controlled))]

    # print results
    for i in range(len(idx)):
        print(f"\ncase {i}:\n-------")
        print(f"\tcl_mean = {round(controlled[i]['cl'][idx[i]:].mean(), 3)}, "
              f"cl_std = {round(controlled[i]['cl'][idx[i]:].std(), 3)}")
        print(f"\tcd_mean = {round(controlled[i]['cd'][idx[i]:].mean(), 3)}, "
              f"cd_std = {round(controlled[i]['cd'][idx[i]:].std(), 3)}")
