"""
get the amount of discarded trajectories from logfile and plot them (rotatingCylinder2D)
"""
import torch as pt
import matplotlib.pyplot as plt

from glob import glob
from os.path import join
from os import path, makedirs


def load_amount_mb_trajectories_and_discards(load_path: str) -> pt.Tensor:
    """
    get the amount of model-generated trajectories and amount of discarded trajectories from log file

    :param load_path: path to the top-level directory containing all trainings of one setup / case
    :return: amount of discarded trajectories and amount of MB-trajectories in total
    """
    n_failed, n_trajectories = [], []

    for seed in sorted(glob(join(load_path, "*log*")), key=lambda x: int(x.split(".")[-2][-1])):
        counter, counter_failed = 0.0, 0.0          # need to be a float in order to compute mean later
        with open(seed, "r") as f:
            data = f.readlines()

        for line in data:
            # data was created without logger, but if training run with current version -> logger
            if line.startswith("discarding trajectory") or line.startswith("INFO:root:discarding trajectory"):
                counter_failed += 1
            try:
                tmp = line.split(",")[0]
            except IndexError:
                tmp = " "

            if tmp.endswith("'execute_prediction.sh'"):
                counter += 1

        n_failed.append(counter_failed)
        n_trajectories.append(counter)
    return pt.tensor([n_failed, n_trajectories])


def plot_discarded_trajectories(discards, xtick_list, env: str = "rotatingCylinder2D") -> None:
    """
    plots the amount of discarded trajectories for a series of different cases

    :param discards: list with the amount of discarded trajectories for each case
    :param xtick_list: list containing the labels for the x-axis
    :param env: either 'rotatingCylinder2D' or 'rotatingPinball2D'
    :return: None
    """
    # create directory for plots
    if not path.exists(join("..", "plots", env)):
        makedirs(join("..", "plots", env))

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    # plot the amount of discards
    fig, ax = plt.subplots(figsize=(6, 2))
    for i, case in enumerate(discards):
        if i == 0:
            ax.scatter(pt.ones((case.size()[1],)) * i, case[0, :], color="black", label="seed no.", marker="o",
                       alpha=0.4)
            ax.scatter(i, pt.mean(case, dim=1)[0], color="red", marker="x", label="mean discards")
        else:
            ax.scatter(pt.ones((case.size()[1],)) * i, case[0, :], color="black", marker="o", alpha=0.4)
            ax.scatter(i, pt.mean(case, dim=1)[0], color="red", marker="x")

    ax.set_xticks(range(len(xtick_list)), xtick_list)
    ax.set_ylabel("$N_{d}$")
    fig.tight_layout()
    plt.legend(loc="upper right", framealpha=1.0, ncol=2, fontsize=10)
    fig.subplots_adjust(wspace=0.25, top=0.98)
    plt.savefig(join("..", "plots", env, "discarded_trajectories.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


if __name__ == "__main__":
    # cases of the rotatingCylinder2D
    cases_cylinder = ["e200_r10_b10_f8_MB_1model", "e200_r10_b10_f8_MB_5models_thr3", "e200_r10_b10_f8_MB_5models_thr2",
                      "e200_r10_b10_f8_MB_10models_thr6", "e200_r10_b10_f8_MB_10models_thr5",
                      "e200_r10_b10_f8_MB_10models_thr3"]
    labels_cylinder = ["$N_{m} = 1$", "$N_{m} = 5$\n$N_{thr} = 3$", "$N_{m} = 5$\n$N_{thr} = 2$",
                       "$N_{m} = 10$\n$N_{thr} = 6$", "$N_{m} = 10$\n$N_{thr} = 5$", "$N_{m} = 10$\n$N_{thr} = 3$"]

    # cases of the rotatingPinball2D
    cases_pinball = ["e150_r10_b10_f300_MB_1model", "e150_r10_b10_f300_MB_5models_thr2",
                     "e150_r10_b10_f300_MB_10models_thr5"]
    labels_pinball = ["$N_{m} = 1$", "$N_{m} = 5$\n$N_{thr} = 2$", "$N_{m} = 10$\n$N_{thr} = 5$"]

    # load the model-generated and discarded trajectories from the log files
    discards_cylinder = [load_amount_mb_trajectories_and_discards(join("..", "data", "rotatingCylinder2D", c)) for c in
                         cases_cylinder]

    discards_pinball = [load_amount_mb_trajectories_and_discards(join("..", "data", "rotatingPinball2D", c)) for c in
                        cases_pinball]

    # plot the results
    plot_discarded_trajectories(discards_cylinder, labels_cylinder)
    plot_discarded_trajectories(discards_pinball, labels_pinball, env="rotatingPinball2D")
