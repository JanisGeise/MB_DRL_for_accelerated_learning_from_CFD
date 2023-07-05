"""
 load and plot the results of th training for the 'rotatingPinball2D' environment
"""
import pandas as pd
import torch as pt
import matplotlib.pyplot as plt

from glob import glob
from os.path import join
from os import path, makedirs

from plot_results_cylinder import plot_rewards_vs_episode, plot_coefficients_vs_episode, plot_cl_cd_alpha_beta


def load_rewards(load_path: str) -> dict:
    """
    loads the observations_*.pt files and computes the mean and std. deviation of the results wrt episodes

    :param load_path: path to the top-level directory of the case for which the results should be loaded
    :return: dict containing the mean rewards, cl & cd values along with their corresponding std. deviation
    """
    files = sorted(glob(join(load_path, "observations_*.pt")), key=lambda x: int(x.split("_")[-1].split(".")[0]))
    obs = [pt.load(join(f)) for f in files]
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

    # get the model-free episodes
    obs_out["MF_episodes"] = [i for i, o in enumerate(obs) if not "generated_by" in o[0]]

    return obs_out


def resort_results(data):
    """
    resort the loaded results from list(dict) to dict(list) in order to plot the results easier / more efficient
    :param data: the loaded results from the trainings
    :return: the resorted data
    """
    data_out = {}
    for key in list(data[0].keys()):
        data_out[key] = [i[key] for i in results]

    return data_out


def read_forces_for_each_cylinder(load_path: str) -> list:
    """
    load the cl & cd values for each cylinder of the simulations using the final policy

    :param load_path: path to the top-level directory of the simulation conducted using e.g. the final policy
    :return: list with a dataframe containing [t, cd, cl] for each cylinder
    """
    files, data = ["field_cylinder_a", "field_cylinder_b", "field_cylinder_c"], []
    for f in files:
        data.append(pd.read_csv(join(load_path, "postProcessing", f, "0", "surfaceFieldValue.dat"), skiprows=4,
                    header=0, sep=r"\s+", usecols=[0, 1, 2], names=["t", "cd", "cl"]))

    # the x-component has a parenthesis (components are written out as: (Fx, Fy, Fz)), so remove the parenthesis
    for i in range(len(data)):
        data[i]["cd"] = data[i]["cd"].str.replace("(", "", regex=True).astype(float)

    return data


def plot_coefficients_for_each_cylinder(data_controlled: list, data_uncontrolled: list, legend_entries: list,
                                        factor: int = 10) -> None:
    """
    plot cl & cd of the final policies wrt the cylinder and time

    :param data_controlled: list containing the data of the uncontrolled flow
    :param data_uncontrolled: list containing the data of the controlled flow for each case
    :param legend_entries: list containing the legend entries
    :param factor: factor for converting the physical time into non-dimensional time, t^* = u * t / d
    :return: None
    """
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(6, 8), sharex="col")

    # join the data of uncontrolled & controlled in order to plot it easier
    data = [data_uncontrolled] + data_controlled

    # use default color cycle and black for the uncontrolled case
    color = ["black", '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    for row in range(3):
        for col in range(2):
            for c in range(len(data)):
                if col == 0 and row == 0:
                    ax[row][col].plot(data[c][row]["t"] * factor, data[c][row]["cl"], label=legend_entries[c],
                                      color=color[c])
                    ax[row][col].set_ylabel("$c_{L," + f"{row + 1}" + "}$")
                elif col == 0:
                    ax[row][col].plot(data[c][row]["t"] * factor, data[c][row]["cl"], color=color[c])
                    ax[row][col].set_ylabel("$c_{L," + f"{row + 1}" + "}$")
                else:
                    ax[row][col].plot(data[c][row]["t"] * factor, data[c][row]["cd"], color=color[c])
                    ax[row][col].set_ylabel("$c_{D," + f"{row + 1}" + "}$")

    ax[-1][0].set_xlabel("$t^*$")
    ax[-1][1].set_xlabel("$t^*$")
    ax[-1][1].set_xlim(0, 500 * factor)
    ax[-1][0].set_xlim(0, 500 * factor)
    fig.tight_layout()
    fig.legend(loc="upper center", framealpha=1.0, ncol=3)
    fig.subplots_adjust(wspace=0.4, top=0.92)
    plt.savefig(join("..", "plots", "rotatingPinball2D", "cl_cd_vs_cylinder_final_policy.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


if __name__ == "__main__":
    load_path = join("..", "data", "rotatingPinball2D")
    case_name = ["e150_r10_b10_f300_MF", "e150_r10_b10_f300_MB_1model", "e150_r10_b10_f300_MB_5models_thr2",
                 "e150_r10_b10_f300_MB_10models_thr5"]
    legend = ["MF", "MB, $N_{m} = 1$", "MB, $N_{m} = 5, N_{thr} = 2$", "MB, $N_{m} = 10, N_{thr} = 5$"]

    # create directory for plots
    if not path.exists(join("..", "plots", "rotatingPinball2D")):
        makedirs(join("..", "plots", "rotatingPinball2D"))

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    # load rewards
    results = [load_rewards(join(load_path, c, "seed0")) for c in case_name]

    # re-arrange for easier plotting
    results = resort_results(results)

    # print the amount of MF-episodes
    for i, e in enumerate(results["MF_episodes"]):
        print(f"found {len(e)} CFD episodes (= {round(len(e) / len(results['rewards_mean'][i]) * 100, 2)} %)")

    # plot the rewards without std.
    plot_rewards_vs_episode(results["rewards_mean"], n_cases=len(case_name), mf_episodes=results["MF_episodes"],
                            env="rotatingPinball2D", legend=legend)
    """
    # plot the rewards with std. deviation (if needed)
    plot_rewards_vs_episode(results["rewards_mean"], results["rewards_std"], n_cases=len(setup["case_name"]),
                            mf_episodes=results["MF_episodes"], env="rotatingPinball2D", legend=legend)
    """

    # plot cl & cd wrt episode
    plot_coefficients_vs_episode(results["cd_mean"], results["cd_std"], results["cl_mean"], results["cl_std"],
                                 ylabel=["$| \sum \\bar{c}_{L, i} |$", "$| \sum \\bar{c}_{D, i} |$"],
                                 n_cases=len(case_name), env="rotatingPinball2D", legend=legend)

    # get the sum of cd and cl values for all cases
    uncontrolled = pd.read_csv(join(load_path, "uncontrolled", "postProcessing", "forces", "0", "coefficient.dat"),
                               skiprows=12, header=0, sep=r"\s+", usecols=[0, 1, 2], names=["t", "cd", "cl"])

    # then load the sum of cl and cd values for the controlled cases
    controlled = []
    for case in case_name:
        # load the trajectories of the cl and cd coefficients
        controlled.append(pd.read_csv(join(load_path, case, "results_final_policy", "postProcessing", "forces", "0",
                                           "coefficient.dat"), skiprows=12, header=0, sep=r"\s+", usecols=[0, 1, 2],
                                      names=["t", "cd", "cl"]))

    # plot the sum of the cl & cd values of the final policy
    plot_cl_cd_alpha_beta(controlled, uncontrolled, plot_coeffs=True, legend=legend, n_cases=len(case_name),
                          control_start=200, env="rotatingPinball2D")

    # now read forces for each cylinder and plot them (overwrite 'uncontrolled' & 'controlled' since not used anymore)
    uncontrolled = read_forces_for_each_cylinder(join(load_path, "uncontrolled"))

    # then do the same thing for the controlled cases
    controlled = []
    for case in case_name:
        # load the trajectories of the cl and cd coefficients
        controlled.append(read_forces_for_each_cylinder(join(load_path, case, "results_final_policy")))

    # finally plot the forces for each case wrt the cylinder
    plot_coefficients_for_each_cylinder(controlled, uncontrolled, ["uncontrolled"] + legend)
