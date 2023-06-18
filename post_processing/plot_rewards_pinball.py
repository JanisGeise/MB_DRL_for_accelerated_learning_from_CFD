"""
 load and plot the rewards for the 'rotatingPinball2D'
"""
import torch as pt
import matplotlib.pyplot as plt

from glob import glob
from os.path import join
from os import path, mkdir

from plot_ppo_results import plot_rewards_vs_episode


def load_rewards(load_path: str) -> list:
    files = sorted(glob(join(load_path, "observations_*.pt")), key=lambda x: int(x.split("_")[-1].split(".")[0]))
    obs = [pt.load(join(load_path, f)) for f in files]
    r = []
    for episode in range(len(obs)):
        r.append(pt.cat([i["rewards"].unsqueeze(-1) for i in obs[episode]], dim=1))

    # we want the rewards avg. wrt episode, so it's ok to stack all trajectories in the 2nd dimension, since we avg.
    # over all of them anyway
    return [pt.flatten(i) for i in r]


if __name__ == "__main__":
    # Setup
    setup = {
        "main_load_path": r"/home/janis/Hiwi_ISM/results_drlfoam_MB/",
        "path_controlled": r"run/final_routine_pinball_AWS/",
        "case_name": ["e50_r10_b10_f300_MB_5models_1st_e_CFD", "e50_r10_b10_f300_MB_5models_1st_2e_CFD"],
        "color": ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf'],  # default color cycle
        "legend": ["MB, $N_{m} = 5$, 1st episode CFD", "MB, $N_{m} = 5$, 1st 2 episodes CFD"]
    }
    # create directory for plots
    if not path.exists(setup["main_load_path"] + setup["path_controlled"] + "plots"):
        mkdir(setup["main_load_path"] + setup["path_controlled"] + "plots")

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    # load rewards
    rewards = []
    for c in setup["case_name"]:
        rewards.append(load_rewards(join(setup["main_load_path"], setup["path_controlled"], c, "seed0")))

    # compute mean and std wrt episode
    r_mean = [pt.tensor([pt.mean(i).item() for i in case]) for case in rewards]
    r_std = [pt.tensor([pt.std(i).item() for i in case]) for case in rewards]

    # plot the rewards
    plot_rewards_vs_episode(setup, r_mean, r_std, n_cases=2)


