"""
get the amount of discarded trajectories from logfile and plot them
"""
import torch as pt
import matplotlib.pyplot as plt

from glob import glob
from os.path import join


def get_number_of_discards_and_total_trajectories(path: str) -> pt.Tensor:
    n_failed, n_trajectories = [], []

    for seed in sorted(glob(path + "*.log"), key=lambda x: int(x.split(".")[0][-1])):
        counter, counter_failed = 0.0, 0.0          # need to float in order to compute mean later
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


if __name__ == "__main__":
    setup = {
        "path": r"/home/janis/Hiwi_ISM/results_drlfoam_MB/run/final_routine_AWS/",
        "cases": ["e200_r10_b10_f8_MB_1model/", "e200_r10_b10_f8_MB_5models/", "e200_r10_b10_f8_MB_5models_threshold40/",
                  "e200_r10_b10_f8_MB_10models/", "e200_r10_b10_f8_MB_10models_threshold50/",
                  "e200_r10_b10_f8_MB_10models_threshold30/"],
        "n_traj_MF": 10000,     # 5 seeds * 200 episodes * buffer_size
        "labels": ["$N_{m} = 1$", "$N_{m} = 5$\n$N_{thr} = 3$", "$N_{m} = 5$\n$N_{thr} = 2$",
                   "$N_{m} = 10$\n$N_{thr} = 6$", "$N_{m} = 10$\n$N_{thr} = 5$", "$N_{m} = 10$\n$N_{thr} = 3$"],
    }
    n_discards = []

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    for c in setup["cases"]:
        n_discards.append(get_number_of_discards_and_total_trajectories(join(setup["path"], c)))

    # plot the amount of discards
    fig, ax = plt.subplots(figsize=(6, 2))
    for i, case in enumerate(n_discards):
        if i == 0:
            ax.scatter(pt.ones((case.size()[1],)) * i, case[0, :], color="black", label="seed no.", marker="o",
                       alpha=0.4)
            ax.scatter(i, pt.mean(case, dim=1)[0], color="red", marker="x", label="mean discards")
        else:
            ax.scatter(pt.ones((case.size()[1],)) * i, case[0, :], color="black", marker="o", alpha=0.4)
            ax.scatter(i, pt.mean(case, dim=1)[0], color="red", marker="x")

    ax.set_xticks(range(len(setup["labels"])), setup["labels"])
    ax.set_ylabel("$N_{d}$")
    fig.tight_layout()
    plt.legend(loc="upper right", framealpha=1.0, ncol=2, fontsize=10)
    fig.subplots_adjust(wspace=0.25, top=0.98)
    plt.savefig(join(setup["path"], "plots", "discarded_trajectories.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")
