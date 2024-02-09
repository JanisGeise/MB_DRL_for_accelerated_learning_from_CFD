"""
    loads flow fields of 'rotatingCylinder2D' and 'rotatingPinball2D' and compares the mean and standard deviation wrt
    time for different policies and the uncontrolled flow
"""
import torch as pt

from os.path import join
from os import path, makedirs
from typing import Tuple, Union
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from flowtorch.data import FOAMDataloader, mask_box


def load_pressure_fields(load_dir: str, boundaries: list = None, start_time: Union[int, float] = 8,
                         field_name: str = "p") -> Tuple[pt.Tensor, pt.Tensor]:
    """
    load the pressure field of the cylinder2D case, mask out an area

    :param load_dir: path to the simulation data
    :param boundaries: list with list containing the upper and lower boundaries of the mask as:
                       [[xmin, ymin], [xmax, ymax]]
    :param start_time: first time step for which the flow field is loaded
    :param field_name: name of the field which should be loaded
    :return: pressure fields at each write time, x- & y-coordinates of the cells as tuples
    """
    # in case there are no boundaries specified, use the complete domain
    boundaries = [[-1e6, -1e6], [1e6, 1e6]] if boundaries is None else boundaries

    # create foam loader object
    loader = FOAMDataloader(load_dir)

    # load vertices and discard z-coordinate, since both cases are 2D
    vertices = loader.vertices[:, :2]
    mask = mask_box(vertices, lower=boundaries[0], upper=boundaries[1])

    # assemble data matrix
    write_time = [t for t in loader.write_times[1:] if float(t) >= start_time]
    data = pt.zeros((mask.sum().item(), len(write_time)), dtype=pt.float32)
    for i, t in enumerate(write_time):
        # load the specified field
        data[:, i] = pt.masked_select(loader.load_snapshot(field_name, t), mask)

    # stack the coordinates to tuples
    xy = pt.stack([pt.masked_select(vertices[:, 0], mask), pt.masked_select(vertices[:, 1], mask)], dim=1)

    return data, xy


def plot_flow_fields(coord: Union[list, Tuple], field_values: Union[list, Tuple], simulation: str = "cylinder") -> None:
    if simulation == "cylinder":
        # diameter of cylinders (same for all cases)
        d = 0.1
        target_dir = "rotatingCylinder2D"
        fig_size = (8, 3)
    else:
        d = 1
        target_dir = "rotatingPinball2D"
        fig_size = (8, 6)

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    # create directory for plots
    if not path.exists(join("..", "plots", target_dir)):
        makedirs(join("..", "plots", target_dir))

    fig, ax = plt.subplots(ncols=2, nrows=len(cases_cylinder), figsize=fig_size, sharex="all", sharey="row")

    for c in range(len(cases_cylinder)):
        ax[c][0].tricontourf(coord[c][:, 0] / d, coord[c][:, 1] / d, pt.mean(field_values[c], 1))
        ax[c][1].tricontourf(coord[c][:, 0] / d, coord[c][:, 1] / d, pt.std(field_values[c], 1))
        for i in range(2):
            if simulation == "cylinder":
                ax[c][i].add_patch(Circle((0.2 / d, 0.2 / d), (d / 2) / d, facecolor="white"))
            else:
                ax[c][i].add_patch(Circle((-1.299 / d, 0), (d / 2) / d, facecolor="white"))
                ax[c][i].add_patch(Circle((0, 0.75 / d), (d / 2) / d, facecolor="white"))
                ax[c][i].add_patch(Circle((0, -0.75 / d), (d / 2) / d, facecolor="white"))

            ax[-1][i].set_xlabel("$x / d$")
            ax[c][i].set_aspect("equal")
        ax[c][0].set_ylabel("$y / d$")
    ax[0][0].set_title(r"$\mu (p)$")
    ax[0][1].set_title(r"$\sigma (p)$")
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.15)
    plt.savefig(join("..", "plots", target_dir, "comparison_pressure_fields.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


if __name__ == "__main__":
    # path to the top-level directory
    path_cylinder = join("..", "data", "rotatingCylinder2D")
    path_pinball = join("..", "data", "rotatingPinball2D")

    # simulations which should be compared
    cases_cylinder = ["uncontrolled", join("e200_r10_b10_f8_MF", "seed4", "results_best_policy"),
                      join("e200_r10_b10_f8_MB_10models_thr5", "seed0", "results_best_policy")]

    cases_pinball = ["uncontrolled", join("e150_r10_b10_f300_MF", "results_final_policy"),
                     join("e150_r10_b10_f300_MB_10models_thr5", "results_final_policy")]

    # load the CFD data (here only the field required) and compute mean and std. deviation wrt time
    pressure, coordinates = zip(*[load_pressure_fields(join(path_cylinder, c)) for c in cases_cylinder])

    # plot the pressure fields for the cylinder simulation
    plot_flow_fields(coordinates, pressure)

    # now load the data from the pinball simulations
    pressure, coordinates = zip(*[load_pressure_fields(join(path_pinball, c), start_time=350) for c in cases_pinball])

    # plot the pressure fields for the pinball simulation
    plot_flow_fields(coordinates, pressure, simulation="pinball")
