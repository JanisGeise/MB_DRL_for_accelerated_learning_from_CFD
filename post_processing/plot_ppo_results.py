"""
    brief:
        - post-processes and plots the results of the training using PPO, either for MF-DRL or MB-DRL
        - plots the results of the controlled case using the best policy in comparison to the uncontrolled case

    dependencies:
        - 'ppo_data_loader.py' for handling the loading, sorting and merging of all training data
        - 'analyze_frequency_spectrum.py' for plotting the frequency spectrum of cl- and cd of PPO-training and the
           final results

    prerequisites:
        - execution of the 'run_training_MB.py' function in the 'test_training' directory in order to conduct a training
          and generate trajectories within the CFD environment (https://github.com/OFDataCommittee/drlfoam)
        - this training can either be model-free or model-based

    optional:
        - execution of simulation for the best policy from training, also results of a simulation without control
        - in this case, the results of the training are not averaged over multiple seed values, since the final policy
          corresponds to a specific training of one seed value
"""
import pandas as pd
import matplotlib.pyplot as plt

from os.path import join
from typing import Union
from os import mkdir, path

from ppo_data_loader import *


def plot_results_vs_episode(settings: dict, cd_mean: Union[list, pt.Tensor], cd_std: Union[list, pt.Tensor],
                            cl_mean: Union[list, pt.Tensor], cl_std: Union[list, pt.Tensor],
                            actions_mean: Union[list, pt.Tensor], actions_std: Union[list, pt.Tensor],
                            n_cases: int = 1, plot_action: bool = True) -> None:
    """
    plot cl, cd and actions (if specified) depending on the episode (training)

    :param settings: dict containing all the paths etc.
    :param cd_mean: mean cd received over the training periode
    :param cd_std: corresponding standard deviation of cd throughout the training periode
    :param cl_mean: mean cl received over the training periode
    :param cl_std: corresponding standard deviation of cl throughout the training periode
    :param actions_mean: mean actions (omega) done over the training periode
    :param actions_std: corresponding standard deviation of the actions done over the training periode
    :param n_cases: number of cases to compare (= number of imported data)
    :param plot_action: if 'True' cl, cd and actions will be plotted, otherwise only cl and cd will be plotted
    :return: None
    """
    if plot_action:
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 5))
        n_subfig = 3
    else:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
        n_subfig = 2

    for c in range(n_cases):
        for i in range(n_subfig):
            if i == 0:
                ax[i].plot(range(len(cl_mean[c])), cl_mean[c], color=settings["color"][c], label=settings["legend"][c])
                ax[i].fill_between(range(len(cl_mean[c])), cl_mean[c] - cl_std[c], cl_mean[c] + cl_std[c],
                                   color=settings["color"][c], alpha=0.3)
                ax[i].set_ylabel("$\\bar{c}_L$")

            elif i == 1:
                ax[i].plot(range(len(cd_mean[c])), cd_mean[c], color=settings["color"][c])
                ax[i].fill_between(range(len(cd_mean[c])), cd_mean[c] - cd_std[c], cd_mean[c] + cd_std[c],
                                   color=settings["color"][c], alpha=0.3)
                ax[i].set_ylabel("$\\bar{c}_D$")

            elif plot_action:
                ax[i].plot(range(len(actions_mean[c])), actions_mean[c], color=settings["color"][c])
                ax[i].fill_between(range(len(actions_mean[c])), actions_mean[c] - actions_std[c],
                                   actions_mean[c] + actions_std[c], color=settings["color"][c], alpha=0.3)
                ax[i].set_ylabel("$\\bar{\omega}$")

            ax[i].set_xlabel("$e$")

    fig.tight_layout()
    fig.legend(loc="upper right", framealpha=1.0, ncol=3)
    fig.subplots_adjust(wspace=0.25, top=0.93)
    plt.savefig(join(settings["main_load_path"], setup["path_controlled"], "plots", "coefficients_vs_episode.png"),
                dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def plot_rewards_vs_episode(settings: dict, reward_mean: Union[list, pt.Tensor], reward_std: Union[list, pt.Tensor],
                            n_cases: int = 0) -> None:
    """
    plots the mean rewards received throughout the training periode and the corresponding standard deviation

    :param settings: dict containing all the paths etc.
    :param reward_mean: mean rewards received over the training periode
    :param reward_std: corresponding standard deviation of the rewards received over the training periode
    :param n_cases: number of cases to compare (= number of imported data)
    :return: None
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    for c in range(n_cases):
        ax.plot(range(len(reward_mean[c])), reward_mean[c], color=settings["color"][c], label=settings["legend"][c])
        ax.fill_between(range(len(reward_mean[c])), reward_mean[c] - reward_std[c], reward_mean[c] + reward_std[c],
                        color=settings["color"][c], alpha=0.3)

    ax.set_ylabel("$\\bar{r}$")
    ax.set_xlabel("$e$")
    ax.set_xlim(0, 200)
    fig.tight_layout()
    ax.legend(loc="lower right", framealpha=1.0, ncol=1)
    fig.subplots_adjust(wspace=0.2)
    plt.savefig(join(settings["main_load_path"], setup["path_controlled"], "plots", "rewards_vs_episode.png"),
                dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def plot_cl_cd_alpha_beta(settings: dict, controlled_cases: Union[list, pt.Tensor],
                          uncontrolled_case: Union[list, pt.Tensor] = None, plot_coeffs=True) -> None:
    """
    plot either cl and cd vs. time or alpha and beta vs. time

    :param settings: dict containing all the paths etc.
    :param controlled_cases: results from the loaded cases with active flow control
    :param uncontrolled_case: reference case containing results from uncontrolled flow past cylinder
    :param plot_coeffs: 'True' means cl and cd will be plotted, otherwise alpha and beta will be plotted wrt to time
    :return: None
    """
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6, 6), sharex="col")

    if plot_coeffs:
        keys = ["t", "cl", "cd"]
        save_name = "comparison_cl_cd"
        ax[1].set_ylim(2.95, 3.25)  # Re = 100
        # ax[1].set_ylim(2, 3.75)          # Re = 500
        n_cases = range(len(settings["case_name"]) + 1)
        ylabels = ["$c_L$", "$c_D$"]
    else:
        keys = ["t", "alpha", "beta"]
        save_name = "comparison_alpha_beta"
        ylabels = ["$\\alpha$", "$\\beta$"]
        n_cases = range(1, len(settings["case_name"]) + 1)

    for c in n_cases:
        for i in range(2):
            if i == 0:
                if c == 0:
                    ax[i].plot(uncontrolled_case[keys[0]] * 10, uncontrolled_case[keys[1]], color="black",
                               label="uncontrolled")
                else:
                    ax[i].plot(controlled_cases[c - 1][keys[0]] * 10, controlled_cases[c - 1][keys[1]],
                               color=settings["color"][c - 1], label=settings["legend"][c - 1])
                ax[i].set_ylabel(ylabels[0], fontsize=14)
            else:
                if c == 0:
                    ax[i].plot(uncontrolled_case[keys[0]] * 10, uncontrolled_case[keys[2]], color="black")
                else:
                    ax[i].plot(controlled_cases[c - 1][keys[0]] * 10, controlled_cases[c - 1][keys[2]],
                               color=settings["color"][c - 1])
                ax[i].set_ylabel(ylabels[1], fontsize=14)

            ax[1].set_xlabel("$t^*$", fontsize=14)
            ax[i].set_xlim(0, 200)
    fig.tight_layout()
    fig.legend(loc="upper center", framealpha=1.0, fontsize=10, ncol=2)
    fig.subplots_adjust(wspace=0.2, top=0.84)
    plt.savefig(join(settings["main_load_path"], settings["path_controlled"], "plots", f"{save_name}.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def plot_omega(settings: dict, controlled_cases: Union[list, pt.Tensor]) -> None:
    """
    plot omega (actions) vs. time

    :param settings: dict containing all the paths etc.
    :param controlled_cases: results from the loaded cases with active flow control
    :return: None
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    for c in range(len(settings["case_name"])):
        ax.plot(controlled_cases[c]["t"] * 10, controlled_cases[c]["omega"], color=settings["color"][c],
                label=settings["legend"][c])

    ax.set_ylabel("$\omega$")
    ax.set_xlabel("$t^*$")
    fig.tight_layout()
    fig.subplots_adjust(top=0.91)
    plt.legend(loc="upper right", framealpha=1.0, ncol=1)
    plt.savefig(join(settings["main_load_path"], settings["path_controlled"], "plots", "omega_controlled_case.png"),
                dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def plot_variance_of_beta_dist(settings: dict, var_beta_dist: Union[list, pt.Tensor], n_cases: int = 0) -> None:
    """
    plots the mean rewards received throughout the training periode and the corresponding standard deviation

    :param settings: dict containing all the paths etc.
    :param var_beta_dist: computed variance of the beta-function wrt episode
    :param n_cases: number of cases to compare (= number of imported data)
    :return: None
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    for c in range(n_cases):
        ax.plot(range(len(var_beta_dist[c])), var_beta_dist[c], color=settings["color"][c], label=settings["legend"][c])

    ax.set_ylabel("$mean$ $variance$ $of$ $beta-distribution$", fontsize=13)
    ax.set_xlabel("$e$", fontsize=13)
    ax.legend(loc="upper right", framealpha=1.0, fontsize=10, ncol=2)
    plt.savefig(join(settings["main_load_path"], setup["path_controlled"], "plots", "var_beta_distribution.png"),
                dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def plot_cl_cd_trajectories(settings: dict, data: list, number: int, e: int = 1) -> None:
    """
    plots the trajectory of cl and cd for different episodes of the training, meant to use for either comparing MF-
    trajectories to trajectories generated by the environment models or comparing trajectories from environment models
    run with different settings to each other

    :param settings: setup containing all the paths etc.
    :param data: trajectory data to plot
    :param number: number of the trajectory within the data set (either within the episode or in total)
    :param e: episode number
    :return: None
    """
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    epochs = pt.tensor(list(range(len(data[0]["cl"][1, :, number])))) / 10      # TODO: 10 = for Re = 100
    for n in range(len(data)):
        for i in range(3):
            try:
                if i == 0:
                    # 2nd episode is always MB (if MB-DRL was used)
                    ax[i].plot(epochs, data[n]["cl"][e, :, number], color=settings["color"][n],
                               label=f"{settings['legend'][n]}, episode {e + 1}")
                    ax[i].set_ylabel("$c_L$")
                elif i == 1:
                    ax[i].plot(epochs, data[n]["cd"][e, :, number], color=settings["color"][n])
                    ax[i].set_ylabel("$c_D$")
                else:
                    ax[i].plot(epochs, data[n]["actions"][e, :, number], color=settings["color"][n])
                    ax[i].set_ylabel("$omega$")
                ax[i].set_xlabel("$t^*$")
            except IndexError:
                print("omit plotting trajectories of failed cases")
    fig.tight_layout()
    fig.legend(loc="upper right", framealpha=1.0, ncol=2)
    fig.subplots_adjust(wspace=0.25, top=0.90)
    plt.savefig(join(settings["main_load_path"], settings["path_controlled"], "plots",
                     f"comparison_traj_cl_cd_{e}.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def plot_mean_std_trajectories(settings: dict, data: list) -> None:
    """
    plots the trajectory of cl and cd for different episodes of the training, meant to use for either comparing MF-
    trajectories to trajectories generated by the environment models or comparing trajectories from environment models
    run with different settings to each other

    :param settings: setup containing all the paths etc.
    :param data: trajectory data to plot
    :return: None
    """
    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(6, 8), sharey="col", sharex="all")
    epochs = pt.tensor(list(range(len(data[0]["cl"][1, :, 0])))) / 10       # TODO: only for Re = 100
    e = [24, 74, 124, 199]
    for n in range(len(data)):
        for k in range(4):          # k = rows
            for i in range(2):      # i = cols
                if i == 0:
                    mean_tmp = pt.mean(data[n]["cd"][e[k], :, :], dim=1)
                    std_tmp = pt.std(data[n]["cd"][e[k], :, :], dim=1)
                    if k == 0:
                        # 2nd episode is always MB (if MB-DRL was used)
                        ax[k][i].plot(epochs, mean_tmp, color=settings["color"][n], label=settings['legend'][n])
                        ax[k][i].fill_between(epochs, mean_tmp - std_tmp, mean_tmp + std_tmp, color=settings["color"][n],
                                              alpha=0.3)
                    else:
                        ax[k][i].plot(epochs, mean_tmp, color=settings["color"][n])
                        ax[k][i].fill_between(epochs, mean_tmp - std_tmp, mean_tmp + std_tmp, color=settings["color"][n],
                                              alpha=0.3)

                    ax[k][i].set_ylabel("$\\bar{c}_D$")
                else:
                    mean_tmp = pt.mean(data[n]["cl"][e[k], :, :], dim=1)
                    std_tmp = pt.std(data[n]["cl"][e[k], :, :], dim=1)
                    ax[k][i].plot(epochs, mean_tmp, color=settings["color"][n])
                    ax[k][i].fill_between(epochs, mean_tmp - std_tmp, mean_tmp + std_tmp, color=settings["color"][n],
                                          alpha=0.3)
                    if i == 1:
                        ax[k][i].set_ylabel("$\\bar{c}_L$")

                ax[k][i].set_xlim(0, 40)
    ax[-1][0].set_xlabel("$t^*$")
    ax[-1][1].set_xlabel("$t^*$")
    fig.tight_layout()
    fig.legend(loc="upper center", framealpha=1.0, ncol=2)
    fig.subplots_adjust(wspace=0.3, top=0.88)
    plt.savefig(join(settings["main_load_path"], settings["path_controlled"], "plots",
                     "comparison_traj_cd_mean_std.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


if __name__ == "__main__":
    # Setup
    setup = {
        "main_load_path": r"/home/janis/Hiwi_ISM/results_drlfoam_MB/",
        "path_to_probes": r"postProcessing/probes/0/p",  # path to the file containing trajectories of probes
        "path_uncontrolled": r"run/uncontrolled_re100/",  # path to uncontrolled reference case
        "path_controlled": r"run/final_routine_AWS/",
        "path_final_results": r"results_best_policy/",  # path to the results using the best policy
        # "case_name": ["e200_r10_b10_f8_MF/", "e200_r10_b10_f8_MB_1model/",
        #              "e200_r10_b10_f8_MB_5models/", "e200_r10_b10_f8_MB_5models_threshold40/",
        #              "e200_r10_b10_f8_MB_10models_threshold50/", "e200_r10_b10_f8_MB_10models_threshold30/"],
        "case_name": ["e200_r10_b10_f8_MF/seed4/", "e200_r10_b10_f8_MB_1model/seed2/",
                      "e200_r10_b10_f8_MB_5models/seed1/", "e200_r10_b10_f8_MB_5models_threshold40/seed1/",
                      "e200_r10_b10_f8_MB_10models_threshold50/seed0/",
                      "e200_r10_b10_f8_MB_10models_threshold30/seed4/"],
        "e_trajectory": [4, 9, 24, 49, 74, 99, 124, 149, 174, 199],   # episodes trajectories (cl & cd, not avg.)
        "n_probes": 12,  # number of probes placed in flow field
        "avg_over_cases": False,  # if cases should be averaged over, e.g. different seeds
        "plot_final_res": False,  # if the final policy already ran in openfoam, plot the results
        "param_study": False,  # flag if parameter study, only used for generating legend entries automatically
        "color": ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf'],  # default color cycle
        "legend": ["MF", "MB, $N_{models} = 1$, every $4^{th}$ episode CFD",
                   "MB, $N_{models} = 5, N_{thr} = 3$", "MB, $N_{models} = 5, N_{thr} = 2$",
                   "MB, $N_{models} = 10, N_{thr} = 5$", "MB, $N_{models} = 10, N_{thr} = 3$"]
    }

    # create directory for plots
    if not path.exists(setup["main_load_path"] + setup["path_controlled"] + "plots"):
        mkdir(setup["main_load_path"] + setup["path_controlled"] + "plots")

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    # load all the data
    all_data = load_all_data(setup)

    # average the trajectories episode-wise
    averaged_data = average_results_for_each_case(all_data)

    # for parameter study: generate legend entries automatically
    if setup["param_study"]:
        if setup["avg_over_cases"]:
            setup["legend"] = [f"b = {averaged_data['buffer_size'][c]}, l = {averaged_data['len_traj'][c]}" for c in
                               range(len(averaged_data["len_traj"]))]

        else:
            setup["legend"] = [f"seed = {c}" for c in range(len(averaged_data["len_traj"]))]

    # print info amount CFD episodes, assuming 1st case is MF
    for i in range(len(averaged_data["MF_episodes"])):
        print(f"{averaged_data['MF_episodes'][i]} CFD episodes for case {i}")

    # plots for avg. the trajectories for cl & cd and plot then vs. t
    plot_mean_std_trajectories(setup, all_data)

    # plot variance of the beta-distribution wrt episodes
    plot_variance_of_beta_dist(setup, averaged_data["var_beta_fct"], n_cases=len(setup["case_name"]))

    # plot mean rewards wrt to episode
    plot_rewards_vs_episode(setup, reward_mean=averaged_data["mean_rewards"], reward_std=averaged_data["std_rewards"],
                            n_cases=len(setup["case_name"]))

    # plot mean cl and cd wrt to episode
    plot_results_vs_episode(setup, cd_mean=averaged_data["mean_cd"], cd_std=averaged_data["std_cd"],
                            cl_mean=averaged_data["mean_cl"], cl_std=averaged_data["std_cl"],
                            actions_mean=averaged_data["mean_actions"], actions_std=averaged_data["std_actions"],
                            n_cases=len(setup["case_name"]), plot_action=False)

    # compare trajectories over the course of the training
    for e in setup["e_trajectory"]:
        plot_cl_cd_trajectories(setup, all_data, number=1, e=e)

    # if the cases are run in openfoam using the trained network (using the best policy), plot the results
    if setup["plot_final_res"]:
        # import the trajectory of the uncontrolled case
        uncontrolled = pd.read_csv(join(setup["main_load_path"], setup["path_uncontrolled"], "postProcessing", "forces",
                                        "0", "coefficient.dat"), skiprows=13, header=0,
                                   sep=r"\s+", usecols=[0, 1, 2], names=["t", "cd", "cl"])
        p_uncontrolled = pd.read_csv("".join([setup["main_load_path"], setup["path_uncontrolled"],
                                              setup["path_to_probes"]]), skiprows=setup["n_probes"] + 1, header=0,
                                     names=["t"] + [f"probe_{i}" for i in range(setup["n_probes"])], sep=r"\s+")

        controlled, p_controlled, traj = [], [], []
        for case in range(len(setup["case_name"])):
            # import the trajectories of the controlled cases
            controlled.append(pd.read_csv(join(setup["main_load_path"], setup["path_controlled"],
                                               setup["case_name"][case], setup["path_final_results"], "postProcessing",
                                               "forces", "0", "coefficient.dat"), skiprows=13, header=0,
                                          sep=r"\s+", usecols=[0, 1, 2], names=["t", "cd", "cl"]))

            traj.append(pd.read_csv(join(setup["main_load_path"], setup["path_controlled"], setup["case_name"][case],
                                         setup["path_final_results"], "trajectory.csv"), header=0, sep=r",",
                                    usecols=[0, 1, 2, 3], names=["t", "omega", "alpha", "beta"]))

            p_controlled.append(pd.read_csv(join(setup["main_load_path"], setup["path_controlled"],
                                                 setup["case_name"][case], setup["path_final_results"],
                                                 setup["path_to_probes"]), skiprows=setup["n_probes"] + 1,
                                            header=0, names=["t"] + [f"probe_{i}" for i in range(setup["n_probes"])],
                                            sep=r"\s+"))

        # plot cl and cd of the controlled cases vs. the uncontrolled cylinder flow
        plot_cl_cd_alpha_beta(setup, controlled, uncontrolled, plot_coeffs=True)

        # plot omega of the controlled cases
        plot_omega(setup, traj)

        # plot alpha and beta of the controlled cases
        plot_cl_cd_alpha_beta(setup, traj, plot_coeffs=False)

