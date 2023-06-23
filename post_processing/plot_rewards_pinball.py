"""
 load and plot the rewards for the 'rotatingPinball2D'
"""
import torch as pt
import matplotlib.pyplot as plt

from glob import glob
from os.path import join
from os import path, mkdir

from plot_ppo_results import plot_rewards_vs_episode, plot_coefficients_vs_episode


def load_rewards(load_path: str) -> dict:
    files = sorted(glob(join(load_path, "observations_*.pt")), key=lambda x: int(x.split("_")[-1].split(".")[0]))
    obs = [pt.load(join(load_path, f)) for f in files]
    obs_out = {"rewards": [], "cl": [], "cd": []}

    for episode in range(len(obs)):
        for key in obs_out:
            if key == "rewards":
                tmp = [i[key].unsqueeze(-1) for i in obs[episode]]
            elif key == "cl":
                tmp = [(i["cy_a"] + i["cy_b"] + i["cy_c"]).abs().unsqueeze(-1) for i in obs[episode]]
            elif key == "cd":
                tmp = [(i["cx_a"] + i["cx_b"] + i["cx_c"]).abs().unsqueeze(-1) for i in obs[episode]]
            else:
                continue
            obs_out[key].append(pt.cat(tmp, dim=1))

    # we want the quantities' avg. wrt episode, so it's ok to stack all trajectories in the 2nd dimension, since we avg.
    # over all of them anyway, list(...) to avoid RuntimeError, bc dict is changing size during iterations
    for key in list(obs_out.keys()):
        obs_out[f"{key}_mean"] = pt.tensor([pt.mean(pt.flatten(i)) for i in obs_out[key]])
        obs_out[f"{key}_std"] = pt.tensor([pt.std(pt.flatten(i)) for i in obs_out[key]])

        # mean & std. values are sufficient, so delete the actual trajectories from dict
        obs_out.pop(key)

    return obs_out


def resort_results(data):
    data_out = {}
    for key in list(data[0].keys()):
        data_out[key] = [i[key] for i in results]

    return data_out


if __name__ == "__main__":
    # Setup
    setup = {
        "main_load_path": r"/home/janis/Hiwi_ISM/results_drlfoam_MB/",
        "path_controlled": r"run/final_routine_pinball_AWS/",
        "case_name": ["test_pinball_lr", "e150_r10_b10_f300_MB_5models_thr40_lr1e-4",
                      "e150_r10_b10_f300_MB_5models_thr40_lr2e-4"],
        "color": ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf'],  # default color cycle
        # "legend": ["MF", "MB, $N_{m} = 1$", "MB, $N_{m} = 5, N_{thr} = 2$", "MB, $N_{m} = 10, N_{thr} = 5$"]
        "legend": ["MF", "MB, $N_{m} = 5, lr=1e-4$", "MB, $N_{m} = 5, lr=2e-4$"]
    }
    # create directory for plots
    if not path.exists(join(setup["main_load_path"], setup["path_controlled"], "plots")):
        mkdir(join(setup["main_load_path"], setup["path_controlled"], "plots"))

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    # load rewards
    results = []
    for c in setup["case_name"]:
        results.append(load_rewards(join(setup["main_load_path"], setup["path_controlled"], c, "seed0")))

    # re-arrange for easier plotting
    results = resort_results(results)

    # plot the rewards
    plot_rewards_vs_episode(setup, results["rewards_mean"], results["rewards_std"], n_cases=len(setup["case_name"]))

    # plot cl & cd wrt episode
    plot_coefficients_vs_episode(setup, results["cd_mean"], results["cd_std"], results["cl_mean"], results["cl_std"],
                                 ylabel=["$| \sum \\bar{c}_{L, i} |$", "$| \sum \\bar{c}_{D, i} |$"],
                                 n_cases=len(setup["case_name"]))
