"""
    - post-processes and plots the results of the PPO-training for the 'rotatingCylinder2D' environment
    - plots the results of the controlled case using the best policy in comparison to the uncontrolled case

    dependencies:
        - 'ppo_data_loader.py' for handling the loading, sorting and merging of all training data
"""
import pandas as pd
import matplotlib.pyplot as plt

from os.path import join
from typing import Union
from os import path, makedirs

from ppo_data_loader import *


def plot_coefficients_vs_episode(cd_mean: Union[list, pt.Tensor], cd_std: Union[list, pt.Tensor],
                                 cl_mean: Union[list, pt.Tensor], cl_std: Union[list, pt.Tensor],
                                 actions_mean: Union[list, pt.Tensor] = None,
                                 actions_std: Union[list, pt.Tensor] = None, n_cases: int = 1,
                                 plot_action: bool = False, ylabel: list = None, env: str = "rotatingCylinder2D",
                                 legend: list = None) -> None:
    """
    plot cl, cd and actions (if specified) depending on the episode (training)

    :param cd_mean: mean cd received over the training periode
    :param cd_std: corresponding standard deviation of cd throughout the training periode
    :param cl_mean: mean cl received over the training periode
    :param cl_std: corresponding standard deviation of cl throughout the training periode
    :param actions_mean: mean actions (omega) done over the training periode
    :param actions_std: corresponding standard deviation of the actions done over the training periode
    :param n_cases: number of cases to compare (= number of imported data)
    :param plot_action: if 'True' cl, cd and actions will be plotted, otherwise only cl and cd will be plotted
    :param ylabel: ylabels for plots, if none then [cl, cd, omega] is used (mean and std.)
    :param env: either 'rotatingCylinder2D' or 'rotatingPinball2D'
    :param legend: list containing the legend entries
    :return: None
    """
    if not ylabel:
        ylabel = [["$\mu(c_L)$", "$\mu(c_D)$", "$\mu(\omega)$"], ["$\sigma(c_L)$", "$\sigma(c_D)$", "$\sigma(\omega)$"]]

    if plot_action:
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(6, 4), sharex="col")
        n_subfig = 3
    else:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(6, 4), sharex="col")
        n_subfig = 2

    # use default color cycle
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    for c in range(n_cases):
        for i in range(n_subfig):
            if i == 0:
                if legend:
                    ax[0][i].plot(range(len(cl_mean[c])), cl_mean[c], color=color[c], label=legend[c])
                    ax[1][i].plot(range(len(cl_std[c])), cl_std[c], color=color[c])
                else:
                    ax[i].plot(range(len(cl_mean[c])), cl_mean[c], color=color[c], label=f"case {c}")
                    ax[1][i].plot(range(len(cl_std[c])), cl_std[c], color=color[c])
                ax[0][i].set_ylabel(ylabel[0][0])
                ax[1][i].set_ylabel(ylabel[1][0])

            elif i == 1:
                ax[0][i].plot(range(len(cd_mean[c])), cd_mean[c], color=color[c])
                ax[1][i].plot(range(len(cd_std[c])), cd_std[c], color=color[c])
                ax[0][i].set_ylabel(ylabel[0][1])
                ax[1][i].set_ylabel(ylabel[1][1])

            elif plot_action:
                ax[0][i].plot(range(len(actions_mean[c])), actions_mean[c], color=color[c])
                ax[1][i].plot(range(len(actions_std[c])), actions_std[c], color=color[c])
                ax[0][i].set_ylabel(ylabel[0][2])
                ax[1][i].set_ylabel(ylabel[1][2])

            ax[1][i].set_xlabel("$e$")
            ax[0][i].set_xlim(0, max([len(i) for i in cl_mean]))
            ax[1][i].set_xlim(0, max([len(i) for i in cl_mean]))

    fig.tight_layout()
    fig.legend(loc="upper center", framealpha=1.0, ncol=2)
    if env == "rotatingCylinder2D":
        fig.subplots_adjust(wspace=0.35, top=0.80)
    else:
        fig.subplots_adjust(wspace=0.35, top=0.86)
    plt.savefig(join("..", "plots", env, "coefficients_vs_episode_.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")
    exit()


def plot_rewards_vs_episode(reward_mean: Union[list, pt.Tensor], reward_std: Union[list, pt.Tensor],
                            n_cases: int = 0, mf_episodes: list = None, env: str = "rotatingCylinder2D",
                            legend: list = None) -> None:
    """
    plots the mean rewards received throughout the training periode and the corresponding standard deviation

    :param reward_mean: mean rewards received over the training periode
    :param reward_std: corresponding standard deviation of the rewards received over the training periode
    :param n_cases: number of cases to compare (= number of imported data)
    :param mf_episodes: list containing information about which episodes are run in CFD
    :param env: either 'rotatingCylinder2D' or 'rotatingPinball2D'
    :param legend: list containing the legend entries
    :return: None
    """
    # use default color cycle
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    fig, ax = plt.subplots(nrows=2, figsize=(6, 4), sharex="col")
    for i in range(2):
        for c in range(n_cases):
            if i == 0:
                if legend:
                    ax[i].plot(range(len(reward_mean[c])), reward_mean[c], color=color[c], label=legend[c])
                else:
                    ax[i].plot(range(len(reward_mean[c])), reward_mean[c], color=color[c], label=f"case {c}")
                ax[i].set_ylabel(r"$\mu(r)$")

            else:
                if legend:
                    ax[i].plot(range(len(reward_std[c])), reward_std[c], color=color[c], label=legend[c])
                else:
                    ax[i].plot(range(len(reward_std[c])), reward_std[c], color=color[c], label=f"case {c}")
                ax[i].set_ylabel(r"$\sigma(r)$")

            ax[i].set_xlim(0, max([len(i) for i in reward_mean]))

            # mark the MF episodes for the 1st training of each case, assuming that the 1st case (c == 0) is model-free
            if mf_episodes and c > 0 and i == 0:
                if c == n_cases - 1:
                    # dummy point for legend, because x all have different colors, but only required once in legend
                    ax[i].scatter(mf_episodes[c], reward_mean[c][pt.tensor(mf_episodes[c])], color="black",
                                  label="$CFD$ $episode$", marker="x")

                    # overwrite the dummy point with actual results, so just the legend entry is black
                    ax[i].scatter(mf_episodes[c], reward_mean[c][pt.tensor(mf_episodes[c])], color=color[c],
                                  marker="x")
                else:
                    ax[i].scatter(mf_episodes[c], reward_mean[c][pt.tensor(mf_episodes[c])], color=color[c],
                                  marker="x")

    ax[1].set_xlabel("$e$")
    fig.tight_layout()
    ax[0].legend(loc="lower right", framealpha=1.0, ncol=2)
    fig.subplots_adjust(wspace=0.2)
    plt.savefig(join("..", "plots", env, "rewards_vs_episode.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def plot_cl_cd_alpha_beta(controlled_cases: Union[list, pt.Tensor],
                          uncontrolled_case: Union[list, pt.Tensor] = None, plot_coeffs=True, factor: int = 10,
                          env: str = "rotatingCylinder2D", control_start: Union[int, float] = 4, legend: list = None,
                          n_cases: int = 0) -> None:
    """
    plot either cl and cd vs. time or alpha and beta vs. time

    :param controlled_cases: results from the loaded cases with active flow control
    :param uncontrolled_case: reference case containing results from uncontrolled flow past cylinder
    :param plot_coeffs: 'True' means cl and cd will be plotted, otherwise alpha and beta will be plotted wrt to time
    :param factor: factor for converting the physical time into non-dimensional time, t^* = u * t / d
    :param env: either 'rotatingCylinder2D' or 'rotatingPinball2D'
    :param control_start: time when the flow control starts
    :param legend: list containing the legend entries
    :param n_cases: number of cases which should be plotted
    :return: None
    """
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6, 6), sharex="col")

    # use default color cycle
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    if plot_coeffs:
        keys = ["t", "cl", "cd"]
        save_name = "comparison_cl_cd"
        n_cases = range(n_cases + 1)
        x_min = 0
        if env == "rotatingCylinder2D":
            ylabels = ["$c_L$", "$c_D$"]
            ax[1].set_ylim(2.95, 3.25)
            top = 0.84
        else:
            ylabels = ["$| \sum \\bar{c}_{L, i} |$", "$| \sum \\bar{c}_{D, i} |$"]
            top = 0.88
    else:
        keys = ["t", "alpha", "beta"]
        save_name = "comparison_alpha_beta"
        ylabels = ["$\\alpha$", "$\\beta$"]
        n_cases = range(1, n_cases + 1)
        x_min = control_start * factor          # there are no alpha & beta available for t < control_start
        top = 0.88

    for c in n_cases:
        for i in range(2):
            if i == 0:
                if c == 0:
                    ax[i].plot(uncontrolled_case[keys[0]] * factor, uncontrolled_case[keys[1]], color="black",
                               label="uncontrolled")
                else:
                    ax[i].plot(controlled_cases[c - 1][keys[0]] * factor, controlled_cases[c - 1][keys[1]],
                               color=color[c - 1], label=legend[c - 1])
                ax[i].set_ylabel(ylabels[0])
            else:
                if c == 0:
                    ax[i].plot(uncontrolled_case[keys[0]] * factor, uncontrolled_case[keys[2]], color="black")
                else:
                    ax[i].plot(controlled_cases[c - 1][keys[0]] * factor, controlled_cases[c - 1][keys[2]],
                               color=color[c - 1])
                ax[i].set_ylabel(ylabels[1])

            ax[1].set_xlabel("$t^*$")
            ax[i].set_xlim(x_min, controlled_cases[0]["t"].iloc[-1] * factor)
    fig.tight_layout()
    fig.legend(loc="upper center", framealpha=1.0, ncol=2)
    fig.subplots_adjust(wspace=0.2, top=top)
    plt.savefig(join("..", "plots", env, f"{save_name}.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def plot_omega(settings: dict, controlled_cases: Union[list, pt.Tensor], factor: int = 10) -> None:
    """
    plot omega (actions) vs. time

    :param settings: dict containing all the paths etc.
    :param controlled_cases: results from the loaded cases with active flow control
    :param factor: factor for converting the physical time into non-dimensional time, t^* = u * t / d
    :return: None
    """
    # use default color cycle
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    for c in range(len(settings["case_name"])):
        ax.plot(controlled_cases[c]["t"] * factor, controlled_cases[c]["omega"], color=color[c],
                label=settings["legend"][c])

    ax.set_ylabel(r"$\omega$")
    ax.set_xlabel("$t^*$")
    fig.tight_layout()
    fig.subplots_adjust()
    plt.legend(loc="upper right", framealpha=1.0, ncol=2)
    plt.savefig(join("..", "plots", "rotatingCylinder2D", "omega_controlled_case.png"), dpi=340)
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
    # use default color cycle
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 3))
    for c in range(n_cases):
        ax.plot(range(len(var_beta_dist[c])), var_beta_dist[c], color=color[c], label=settings["legend"][c])

    ax.set_ylabel("$\mu(\sigma(f(\\alpha, \\beta)$")
    ax.set_xlabel("$e$")
    fig.tight_layout()
    ax.legend(loc="upper right", framealpha=1.0, ncol=2)
    fig.subplots_adjust()
    plt.savefig(join("..", "plots", "rotatingCylinder2D", "var_beta_distribution.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def plot_cl_cd_trajectories(settings: dict, data: list, number: int, e: int = 1, factor: int = 10) -> None:
    """
    plots the trajectory of cl and cd for different episodes of the training, meant to use for either comparing MF-
    trajectories to trajectories generated by the environment models or comparing trajectories from environment models
    run with different settings to each other

    :param settings: setup containing all the paths etc.
    :param data: trajectory data to plot
    :param number: number of the trajectory within the data set (either within the episode or in total)
    :param e: episode number
    :param factor: factor for converting the physical time into non-dimensional time, t^* = u * t / d
    :return: None
    """
    # use default color cycle
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))
    epochs = pt.tensor(list(range(len(data[0]["cl"][1, :, number])))) / factor
    for n in range(len(data)):
        for i in range(3):
            try:
                if i == 0:
                    # 2nd episode is always MB (if MB-DRL was used)
                    ax[i].plot(epochs, data[n]["cl"][e, :, number], color=color[n],
                               label=f"{settings['legend'][n]}, $episode$ ${e + 1}$")
                    ax[i].set_ylabel("$c_L$")
                elif i == 1:
                    ax[i].plot(epochs, data[n]["cd"][e, :, number], color=color[n])
                    ax[i].set_ylabel("$c_D$")
                else:
                    ax[i].plot(epochs, data[n]["actions"][e, :, number], color=color[n])
                    ax[i].set_ylabel(r"$\omega$")
                ax[i].set_xlabel("$t^*$")
            except IndexError:
                print("omit plotting trajectories of failed cases")
    fig.tight_layout()
    fig.legend(loc="upper right", framealpha=1.0, ncol=3)
    fig.subplots_adjust(wspace=0.35, top=0.81)
    plt.savefig(join("..", "plots", "rotatingCylinder2D", f"comparison_traj_cl_cd_{e}.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def plot_mean_std_trajectories(settings: dict, data: list, factor: int = 10) -> None:
    """
    plots the trajectory of cl and cd for different episodes of the training, meant to use for either comparing MF-
    trajectories to trajectories generated by the environment models or comparing trajectories from environment models
    run with different settings to each other

    :param settings: setup containing all the paths etc.
    :param data: trajectory data to plot
    :param factor: factor for converting the physical time into non-dimensional time, t^* = u * t / d
    :return: None
    """
    # use default color cycle
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(6, 8), sharey="col", sharex="all")
    epochs = pt.tensor(list(range(len(data[0]["cl"][1, :, 0])))) / factor
    e = [24, 74, 124, 199]
    for n in range(len(data)):
        for k in range(4):  # k = rows
            for i in range(2):  # i = cols
                if i == 0:
                    mean_tmp = pt.mean(data[n]["cd"][e[k], :, :], dim=1)
                    std_tmp = pt.std(data[n]["cd"][e[k], :, :], dim=1)
                    if k == 0:
                        # 2nd episode is always MB (if MB-DRL was used)
                        ax[k][i].plot(epochs, mean_tmp, color=color[n], label=settings['legend'][n])
                        # ax[k][i].fill_between(epochs, mean_tmp - std_tmp, mean_tmp + std_tmp,
                        #                       color= color[n], alpha=0.3)
                    else:
                        ax[k][i].plot(epochs, mean_tmp, color=color[n])
                        # ax[k][i].fill_between(epochs, mean_tmp - std_tmp, mean_tmp + std_tmp,
                        #                       color= color[n], alpha=0.3)

                    ax[k][i].set_ylabel("$\\bar{c}_D$")
                else:
                    mean_tmp = pt.mean(data[n]["cl"][e[k], :, :], dim=1)
                    std_tmp = pt.std(data[n]["cl"][e[k], :, :], dim=1)
                    ax[k][i].plot(epochs, mean_tmp, color=color[n])
                    # ax[k][i].fill_between(epochs, mean_tmp - std_tmp, mean_tmp + std_tmp, color= color[n],
                    #                       alpha=0.3)
                    if i == 1:
                        ax[k][i].set_ylabel("$\\bar{c}_L$")

                ax[k][i].set_xlim(0, data[0]["cl"].size()[1] / factor)
    ax[-1][0].set_xlabel("$t^*$")
    ax[-1][1].set_xlabel("$t^*$")
    fig.tight_layout()
    fig.legend(loc="upper center", framealpha=1.0, ncol=2)
    fig.subplots_adjust(wspace=0.3, top=0.9)
    plt.savefig(join("..", "plots", "rotatingCylinder2D", "comparison_traj_cd_mean_std.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


if __name__ == "__main__":
    # Setup
    setup = {
        "load_path": join("..", "data", "rotatingCylinder2D"),
        "case_name": ["e200_r10_b10_f8_MF/", "e200_r10_b10_f8_MB_1model/", "e200_r10_b10_f8_MB_5models_thr3",
                      "e200_r10_b10_f8_MB_5models_thr2", "e200_r10_b10_f8_MB_10models_thr5",
                      "e200_r10_b10_f8_MB_10models_thr3"],
        "mark_e_cfd": True,  # flag if CFD episodes should be marked (iCFD episodes of 1st seed are taken)
        "legend": ["MF", "MB, $N_{m} = 1$", "MB, $N_{m} = 5, N_{thr} = 3$", "MB, $N_{m} = 5, N_{thr} = 2$",
                   "MB, $N_{m} = 10, N_{thr} = 5$", "MB, $N_{m} = 10, N_{thr} = 3$"]
    }

    # create directory for plots
    if not path.exists(join("..", "plots", "rotatingCylinder2D")):
        makedirs(join("..", "plots", "rotatingCylinder2D"))

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    # load all the data
    all_data = load_all_data(setup)

    # average the trajectories over all seeds and wrt to episodes
    averaged_data = average_results_for_each_case(all_data)

    # print info amount CFD episodes, assuming 1st case is MF
    for i in range(len(averaged_data["MF_episodes"])):
        print(f"{averaged_data['MF_episodes'][i]} CFD episodes for case {i}")

    # plot mean rewards wrt to episode and the corresponding std. deviation
    if setup["mark_e_cfd"]:
        # take the 1st case and create list with MF episodes
        n_episodes = list(range(averaged_data["mean_rewards"][0].size()[0]))
        e_cfd = [[i for i in n_episodes if i not in case[0]] for case in averaged_data["e_number_mb"]]
        plot_rewards_vs_episode(reward_mean=averaged_data["mean_rewards"], mf_episodes=e_cfd,
                                reward_std=averaged_data["std_rewards"], n_cases=len(setup["case_name"]),
                                legend=setup["legend"])
    else:
        plot_rewards_vs_episode(reward_mean=averaged_data["mean_rewards"],
                                reward_std=averaged_data["std_rewards"], n_cases=len(setup["case_name"]),
                                legend=setup["legend"])

    # avg. the trajectories for cl & cd and plot them wrt t
    plot_mean_std_trajectories(setup, all_data)

    # plot variance of the beta-distribution wrt episodes
    plot_variance_of_beta_dist(setup, averaged_data["var_beta_fct"], n_cases=len(setup["case_name"]))

    # plot mean cl and cd wrt to episode (avg. over t)
    plot_coefficients_vs_episode(cd_mean=averaged_data["mean_cd"], cd_std=averaged_data["std_cd"],
                                 cl_mean=averaged_data["mean_cl"], cl_std=averaged_data["std_cl"],
                                 n_cases=len(setup["case_name"]), plot_action=False, legend=setup["legend"])

    # compare trajectories from defined episodes (qualitatively since trajectories originate from different trainings)
    for e in [4, 9, 24, 49, 74, 99, 124, 149, 174, 199]:
        plot_cl_cd_trajectories(setup, all_data, number=1, e=e)

    # plot the results of the simulations conducted using the best policy
    # first import the cl & cd coefficients of the uncontrolled case
    uncontrolled = pd.read_csv(join(setup["load_path"], "uncontrolled", "postProcessing", "forces", "0",
                                    "coefficient.dat"), skiprows=13, header=0, sep=r"\s+", usecols=[0, 1, 2],
                               names=["t", "cd", "cl"])

    # then load the controlled ones
    controlled, traj = [], []
    for case in setup["case_name"]:
        # get the path to the results using the final policy
        load_path = glob(join(setup["load_path"], case, "*", "results_best_policy"))[0]

        # load the trajectories of the cl and cd coefficients
        controlled.append(pd.read_csv(join(load_path, "postProcessing", "forces", "0", "coefficient.dat"), skiprows=13,
                                      header=0, sep=r"\s+", usecols=[0, 1, 2], names=["t", "cd", "cl"]))

        # load the trajectories of the alpha, beta and omega wrt t
        traj.append(pd.read_csv(join(load_path, "trajectory.csv"), header=0, sep=r",", usecols=[0, 1, 2, 3],
                                names=["t", "omega", "alpha", "beta"]))

    # plot cl and cd of the controlled cases vs. the uncontrolled cylinder flow
    plot_cl_cd_alpha_beta(controlled, uncontrolled, plot_coeffs=True, legend=setup["legend"],
                          n_cases=len(setup["case_name"]))

    # plot omega of the controlled cases
    plot_omega(setup, traj)

    # plot alpha and beta of the controlled cases
    plot_cl_cd_alpha_beta(traj, plot_coeffs=False, legend=setup["legend"], n_cases=len(setup["case_name"]))
