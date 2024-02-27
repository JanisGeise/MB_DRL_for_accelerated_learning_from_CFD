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
    ax.boxplot(discards, labels=xtick_list)
    ax.set_ylabel("$N_\mathrm{d}$")
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.25, top=0.98)
    plt.savefig(join("..", "plots", env, "discarded_trajectories.pdf"), bbox_inches="tight")
    plt.close("all")


if __name__ == "__main__":
    # cases of the rotatingCylinder2D
    cases_cylinder = ["e200_r10_b10_f8_MB_1model", "e200_r10_b10_f8_MB_5models_thr3", "e200_r10_b10_f8_MB_5models_thr2",
                      "e200_r10_b10_f8_MB_10models_thr6", "e200_r10_b10_f8_MB_10models_thr5",
                      "e200_r10_b10_f8_MB_10models_thr3"]
    labels_cylinder = ["$N_\mathrm{m} = 1$", "$N_\mathrm{m} = 5$\n$N_\mathrm{thr} = 3$",
                       "$N_\mathrm{m} = 5$\n$N_\mathrm{thr} = 2$", "$N_\mathrm{m} = 10$\n$N_\mathrm{thr} = 6$",
                       "$N_\mathrm{m} = 10$\n$N_\mathrm{thr} = 5$", "$N_\mathrm{m} = 10$\n$N_\mathrm{thr} = 3$"]

    # cases of the rotatingPinball2D
    cases_pinball = ["e150_r10_b10_f300_MB_1model", "e150_r10_b10_f300_MB_5models_thr2",
                     "e150_r10_b10_f300_MB_10models_thr5"]
    labels_pinball = ["$N_\mathrm{m} = 1$", "$N_\mathrm{m} = 5$\n$N_\mathrm{thr} = 2$",
                      "$N_\mathrm{m} = 10$\n$N_\mathrm{thr} = 5$"]

    # load the model-generated and discarded trajectories from the log files
    discards_cylinder = [load_amount_mb_trajectories_and_discards(join("..", "data", "rotatingCylinder2D", c)) for c in
                         cases_cylinder]

    discards_pinball = [load_amount_mb_trajectories_and_discards(join("..", "data", "rotatingPinball2D", c)) for c in
                        cases_pinball]

    # resort since we don't need the amount of MB episodes here
    discards_cylinder = [d[0, :] for d in discards_cylinder]
    discards_pinball = [d[0, :] for d in discards_pinball]

    # plot the results
    plot_discarded_trajectories(discards_cylinder, labels_cylinder)
    plot_discarded_trajectories(discards_pinball, labels_pinball, env="rotatingPinball2D")
