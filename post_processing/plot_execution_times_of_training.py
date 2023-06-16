"""
get the execution times for CFD, model training, PPO-training, MB-episodes and other from log file & plot the results
"""
import torch as pt
import matplotlib.pyplot as plt

from glob import glob
from os.path import join


def average_exec_times_over_seeds(path: str) -> dict:
    cases = ["time per CFD episode", "time per model training", "time per MB-episode",
             "time per update of PPO-agent", "other"]
    times = {"cfd": [], "mb": [], "model_train": [], "ppo": [], "other": [], "t_total": []}

    for seed in sorted(glob(path + "*.log"), key=lambda x: int(x.split(".")[0][-1])):
        with open(seed, "r") as f:
            data = f.readlines()

        for idx, key in enumerate(times):
            if key == "other":
                tmp = [data[line] for line in range(len(data)) if data[line].startswith(cases[idx])]
                times[key].append(float(tmp[0].split(" ")[1]))

            elif key == "t_total":
                # get the total execution time of training
                times[key].append(float(data[-1].split()[-1].strip("\n")))

            else:
                tmp = [data[line + 5] for line in range(len(data) - 5) if data[line].startswith(cases[idx])]
                times[key].append(float(tmp[0].split(" ")[1]))

    # average exec times over all seeds -> note: due to avg. the times will not exactly add up to 100%
    t = {}
    for key in times:
        t[f"{key}_std"] = pt.std(pt.tensor(times[key]))
        times[key] = pt.mean(pt.tensor(times[key]))
    times.update(t)
    return times


if __name__ == "__main__":
    setup = {
        "path": r"/home/janis/Hiwi_ISM/results_drlfoam_MB/run/final_routine_AWS/",
        "trainings": ["e200_r10_b10_f8_MB_1model/", "e200_r10_b10_f8_MB_5models/",
                      "e200_r10_b10_f8_MB_5models_threshold40/", "e200_r10_b10_f8_MB_10models/",
                      "e200_r10_b10_f8_MB_10models_threshold50/", "e200_r10_b10_f8_MB_10models_threshold30/"],
        "labels": ["$N_{m} = 1$", "$N_{m} = 5$\n$N_{thr} = 3$", "$N_{m} = 5$\n$N_{thr} = 2$",
                   "$N_{m} = 10$\n$N_{thr} = 6$", "$N_{m} = 10$\n$N_{thr} = 5$", "$N_{m} = 10$\n$N_{thr} = 3$"],
        "color": ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf'],       # default color cycle
        "t_MF": 51953.9453  # t_exec of MF case in [s] -> used for scaling
    }
    t_exec = []

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    for t in setup["trainings"]:
        t_exec.append(average_exec_times_over_seeds(join(setup["path"], t)))

    # plot the avg. exec time in [s]
    fig, ax = plt.subplots(figsize=(6, 4))
    legend = ["CFD episode", "MB episode", "model training", "PPO training", "other"]
    for idx, times in enumerate(t_exec):
        bot = 0
        for l, key in enumerate(times):
            if key != "t_total" and key[-4:] != "_std":
                if idx == 0:
                    b = ax.bar(setup["labels"][idx], (times[key] / 100) * times["t_total"] / setup["t_MF"],
                               label=legend[l], bottom=bot, color=setup["color"][l])
                else:
                    b = ax.bar(setup["labels"][idx], (times[key] / 100) * times["t_total"] / setup["t_MF"],
                               bottom=bot, color=setup["color"][l])
                if key != "ppo" and key != "other":
                    t = "{:.2f} \%".format(times[key])
                    ax.bar_label(b, label_type="center", labels=[t], fontsize=7)
                bot += ((times[key] / 100) * times["t_total"] / setup["t_MF"])
    ax.set_ylabel("$t_{exec} \,/\, t_{MF}$", usetex=True)

    fig.tight_layout()
    fig.legend(loc="upper center", framealpha=1.0, ncol=3)
    fig.subplots_adjust(wspace=0.25, top=0.86)
    plt.savefig(join(setup["path"], "plots", "exec_times_absolute.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")