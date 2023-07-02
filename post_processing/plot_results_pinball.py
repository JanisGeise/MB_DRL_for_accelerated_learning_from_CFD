"""
 load and plot the results of th training for the 'rotatingPinball2D' environment
"""
import torch as pt
import matplotlib.pyplot as plt

from glob import glob
from os.path import join
from os import path, makedirs

from plot_results_cylinder import plot_rewards_vs_episode, plot_coefficients_vs_episode


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


def plot_coefficients_of_final_policy():
    pass


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
    # plot the rewards with std. (if needed)
    plot_rewards_vs_episode(results["rewards_mean"], results["rewards_std"], n_cases=len(setup["case_name"]),
                            mf_episodes=results["MF_episodes"], env="rotatingPinball2D", legend=legend)
    """

    # plot cl & cd wrt episode
    plot_coefficients_vs_episode(results["cd_mean"], results["cd_std"], results["cl_mean"], results["cl_std"],
                                 ylabel=["$| \sum \\bar{c}_{L, i} |$", "$| \sum \\bar{c}_{D, i} |$"],
                                 n_cases=len(case_name), env="rotatingPinball2D", legend=legend)
