"""
get the execution times for CFD, model training, PPO-training, MB-episodes and other from log file & plot the results
"""
import torch as pt
import matplotlib.pyplot as plt

from glob import glob
from os.path import join
from os import path, makedirs


def average_exec_times_over_seeds(load_path: str) -> dict:
    """
    load and compute the avg. execution times for MF-, MB-episodes, PPO-update and model-training form log file

    :param load_path: path to the top-level directory containing all trainings for each setup
    :return: dict with execution times for MF-, MB-episode, model-training and PPO-update
    """
    cases = ["time per CFD episode", "time per model training", "time per MB-episode",
             "time per update of PPO-agent", "other"]
    times = {"cfd": [], "mb": [], "model_train": [], "ppo": [], "other": [], "t_total": []}

    # order doesn't matter since we average over all seeds
    for seed in glob(join(load_path, "*log*")):
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


def get_exec_time_of_mf_training(load_path: str) -> float:
    """
    load the execution times for the MF-trainings and compute the avg. execution time

    :param load_path: path to the top-level directory containing all results with the MF-training
    :return: avg. execution time of the MF-training
    """
    # order doesn't matter here, because we want the avg. exec. time
    t = []
    for seed in glob(join(load_path, "*log*")):
        with open(seed, "r") as f:
            lines = f.readlines()

        # the exec. time of the training is located at the end of the log file
        t.append(float(lines[-1].split(" ")[-1].strip("\n")))

    return pt.mean(pt.tensor(t)).item()


def plot_composition_of_exec_time(exec_time: list, t_mf: float, labels: list,
                                  environment: str = "rotatingCylinder2D") -> None:
    """
    plot the decomposition of the execution times of the MB-trainings as a stacked bar chart scaled wrt the execution
    time of the MF-training

    :param exec_time: decomposed execution times of the MB-trainings
    :param t_mf: avg. execution time of the MF-reference training (used for scaling)
    :param labels: names of the cases, which should be displayed at the x-axis
    :param environment: either 'rotatingCylinder2D' or 'rotatingPinball2D'
    :return: None
    """
    # create directory for plots
    if not path.exists(join("..", "plots", environment)):
        makedirs(join("..", "plots", environment))

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    # default color cycle
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    legend = ["CFD episode", "MB episode", "model training", "PPO training", "other"]

    # plot the avg. exec time in [s]
    fig, ax = plt.subplots(figsize=(6, 4))
    for idx, time in enumerate(exec_time):
        bot = 0
        for i, key in enumerate(time):
            if key != "t_total" and key[-4:] != "_std":
                if idx == 0:
                    b = ax.bar(labels[idx], (time[key] / 100) * time["t_total"] / t_mf, label=legend[i], bottom=bot,
                               color=colors[i])
                else:
                    b = ax.bar(labels[idx], (time[key] / 100) * time["t_total"] / t_mf, bottom=bot, color=colors[i])

                # only add label with percentage if > 7%, otherwise area of bar too small in order to see anything
                if time[key] > 7:
                    label = [r"{:.2f} \%".format(time[key])]
                    ax.bar_label(b, label_type="center", labels=label, fontsize=7)
                bot += ((time[key] / 100) * time["t_total"] / t_mf)
    ax.set_ylabel(r"$t_{exec} \,/\, t_{MF}$")

    fig.tight_layout()
    fig.legend(loc="upper center", framealpha=1.0, ncol=3)
    fig.subplots_adjust(wspace=0.25, top=0.86)
    plt.savefig(join("..", "plots", environment, "execution_times.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


if __name__ == "__main__":
    # avg. exec. times of MF-cases in [s], used for scaling the MB-exec. times and can be found at the end of log-file
    t_mf_cylinder = get_exec_time_of_mf_training(join("..", "data", "rotatingCylinder2D", "e200_r10_b10_f8_MF"))
    t_mf_pinball = get_exec_time_of_mf_training(join("..", "data", "rotatingPinball2D", "e150_r10_b10_f300_MF"))

    # x-labels for the plots
    labels_cylinder = ["$N_{m} = 1$", "$N_{m} = 5$\n$N_{thr} = 3$", "$N_{m} = 5$\n$N_{thr} = 2$",
                       "$N_{m} = 10$\n$N_{thr} = 6$", "$N_{m} = 10$\n$N_{thr} = 5$", "$N_{m} = 10$\n$N_{thr} = 3$"]

    labels_pinball = ["$N_{m} = 1$", "$N_{m} = 5$\n$N_{thr} = 2$", "$N_{m} = 10$\n$N_{thr} = 5$"]

    # MB-trainings for rotatingCylinder2D and rotatingPinball2D
    res_cylinder = ["e200_r10_b10_f8_MB_1model", "e200_r10_b10_f8_MB_5models_thr3", "e200_r10_b10_f8_MB_5models_thr2",
                    "e200_r10_b10_f8_MB_10models_thr6", "e200_r10_b10_f8_MB_10models_thr5",
                    "e200_r10_b10_f8_MB_10models_thr3"]

    res_pinball = ["e150_r10_b10_f300_MB_1model", "e150_r10_b10_f300_MB_5models_thr4",
                   "e150_r10_b10_f300_MB_10models_thr5"]

    # get the exec. times from log files
    t_cylinder = [average_exec_times_over_seeds(join("..", "data", "rotatingCylinder2D", t)) for t in res_cylinder]
    t_pinball = [average_exec_times_over_seeds(join("..", "data", "rotatingPinball2D", t)) for t in res_pinball]

    # plot the results for the rotatingCylinder2D
    plot_composition_of_exec_time(t_cylinder, t_mf_cylinder, labels_cylinder)

    # plot the results for the rotatingPinball2D
    plot_composition_of_exec_time(t_pinball, t_mf_pinball, labels_pinball, environment="rotatingPinball2D")
