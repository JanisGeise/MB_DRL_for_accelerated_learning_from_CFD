"""
    plot the numerical setup of the cylinder2D and pinball
"""
import regex as re
import torch as pt

from os.path import join
from os import path, makedirs
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle


def get_probe_locations(load_dir: str) -> pt.Tensor:
    pattern = r"-?\d.\d+ -?\d.\d+ -?\d.\d+"
    with open(join(load_dir, "system", "controlDict"), "r") as f:
        loc = f.readlines()

    # avoid finding other coordinate tuples, which may be present in the controlDict
    idx = [i for i, l in enumerate(loc) if "probeLocations" in l][0]

    # get coordinates of probes, omit appending empty lists and map strings to floats
    coord = [re.findall(pattern, line) for i, line in enumerate(loc) if re.findall(pattern, line) and i > idx]
    return pt.tensor([list(map(float, i[0].split())) for i in coord])


def plot_numerical_setup_cylinder(load_path: str) -> None:
    """
    plot the domain of the cylinder case

    :param load_path: path to the simulation
    :return: None
    """
    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    # create directory for plots
    if not path.exists(join("..", "plots", "rotatingCylinder2D")):
        makedirs(join("..", "plots", "rotatingCylinder2D"))

    # get the probe locations
    pos_probes = get_probe_locations(load_path)

    # get coordinates of domain and cylinder
    with open(join(load_path, "system", "blockMeshDict"), "r") as f:
        loc = f.readlines()

    # structure in blockMeshDict always the same: [lengthX 2.2, lengthY 0.41, cylinderX 0.2, cylinderY 0.2, radius 0.05]
    l, h, pos_x, pos_y, r = [float(loc[i].strip(";\n").split()[1]) for i in range(16, 21)]

    # plot cylinder, probe locations and annotate the domain
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(pos_probes[:, 0], pos_probes[:, 1], linestyle="None", marker="o", color="red", label="probes")

    # dummy point for legend
    ax.scatter(-10, -10, marker="o", color="black", alpha=0.4, label="cylinder")
    circle = Circle((pos_x, pos_y), radius=r, color="black", alpha=0.4)
    rectangle = Rectangle((0, 0), width=l, height=h, edgecolor="black", linewidth=2, facecolor="none")
    ax.add_patch(circle)
    ax.add_patch(rectangle)
    fig.legend(loc="lower center", framealpha=1.0, fontsize=10, ncol=2)
    ax.set_xlim(0, l)
    ax.set_ylim(0, h)
    ax.set_xticks([])
    ax.set_yticks([])

    # annotate inlet & outlet
    plt.arrow(-0.05, -0.05, 0.1, 0.0, color="black", head_width=0.02, clip_on=False)
    plt.arrow(-0.05, -0.05, 0.0, 0.1, color="black", head_width=0.02, clip_on=False)
    plt.arrow(-0.1, h * 2 / 3 + 0.025, 0.075, -0.05, color="black", head_width=0.015, clip_on=False)
    plt.arrow(-0.1 + l, h * 2 / 3, 0.075, -0.05, color="black", head_width=0.015, clip_on=False)

    plt.annotate("$inlet$", (-0.17, h * 2 / 3 + 0.05), annotation_clip=False, fontsize=13)
    plt.annotate("$\\frac{x}{d}$", (0.1, -0.065), annotation_clip=False, fontsize=16)
    plt.annotate("$\\frac{y}{d}$", (-0.1, 0.065), annotation_clip=False, fontsize=16)
    plt.annotate("$outlet$", (-0.2 + l, h * 2 / 3 + 0.01), annotation_clip=False, fontsize=13)

    # annotate the dimensions & position of the domain
    pos = {"xy": [(0, h + 0.04), (0, h), (l, h), (pos_x - r - 0.01, pos_y - 0.1), (l, h), (l, 0), (l + 0.04, h),
                  (pos_x, pos_y + 0.9 * r), (0, pos_y)],
           "xytxt": [(l, h + 0.04), (0, h + 0.075), (l, h + 0.075), (pos_x + r + 0.01, pos_y - 0.1), (l + 0.075, h),
                     (l + 0.075, 0),
                     (l + 0.04, 0), (pos_x, h), (pos_x - 0.9 * r, pos_y)],
           "style": [("<->", "-"), ("-", "--"), ("-", "--"), ("<->", "-"), ("-", "--"), ("-", "--"), ("<->", "-"),
                     ("<->", "-"), ("<->", "-")]
           }
    for i in range(len(pos["style"])):
        plt.annotate("", xy=pos["xy"][i], xytext=pos["xytxt"][i],
                     arrowprops=dict(arrowstyle=pos["style"][i][0], color="black", linestyle=pos["style"][i][1]),
                     annotation_clip=False)

    plt.annotate(f"${l / (2 * r)}$", (l / 2, h + 0.07), annotation_clip=False)
    plt.annotate(f"${h / (2 * r)}$", (l + 0.07, h / 2), annotation_clip=False)
    plt.annotate("$d$", (pos_x - r / 4, pos_y - 3 * r))
    plt.annotate("${:.2f}$".format((h - (pos_y + r)) / (2 * r)), (pos_x + 0.025, pos_y + 2.25 * r))
    plt.annotate("${:.2f}$".format((pos_x - r) / (2 * r)), (pos_x - 3.25 * r, pos_y + 0.5 * r))

    ax.plot((pos_x - r, pos_x - r), (pos_y, pos_y - 0.15), color="black", linestyle="--", lw=1)
    ax.plot((pos_x + r, pos_x + r), (pos_y, pos_y - 0.15), color="black", linestyle="--", lw=1)
    ax.plot(pos_x, pos_y, marker="+", color="black")

    ax.set_aspect("equal")
    fig.tight_layout()
    fig.subplots_adjust(left=0.1, right=0.92)
    plt.savefig(join("..", "plots", "rotatingCylinder2D", "domain_cylinder.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def plot_numerical_setup_pinball(load_path: str) -> None:
    """
    plot the domain of the pinball case

    :param load_path: path to the simulation
    :return: None
    """
    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    # create directory for plots
    if not path.exists(join("..", "plots", "rotatingPinball2D")):
        makedirs(join("..", "plots", "rotatingPinball2D"))

    # get the probe locations
    pos_probes = get_probe_locations(load_path)

    # get coordinates of domain and cylinder
    with open(join(load_path, "system", "blockMeshDict"), "r") as f:
        loc = f.readlines()

    # get domain length and height
    xmin, xmax = float(loc[19].strip(";\n").split()[1].strip(";")), float(loc[20].strip(";\n").split()[1])
    ymin, ymax = float(loc[21].strip(";\n").split()[1].strip(";")), float(loc[22].strip(";\n").split()[1])
    l = xmax - xmin
    h = ymax - ymin

    # get cylinder positions
    pos_x = [float(loc[i].strip(";\n").split()[1]) for i in range(76, 79)]
    pos_y = [float(loc[i].strip(";\n").split()[1]) for i in range(79, 82)]
    r = [float(loc[i].strip(";\n").split()[1]) for i in range(82, 85)]

    # plot cylinder, probe locations and annotate the domain
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(pos_probes[:, 0], pos_probes[:, 1], linestyle="None", marker="o", color="red", label="probes")

    # dummy point for legend
    ax.scatter(-10, -10, marker="o", color="black", alpha=0.4, label="cylinder")
    rectangle = Rectangle((-6, -6), width=l, height=h, edgecolor="black", linewidth=2, facecolor="none")

    # plot cylinders
    for c in range(len(r)):
        circle = Circle((pos_x[c], pos_y[c]), radius=r[c], color="black", alpha=0.4)
        ax.add_patch(circle)
        ax.annotate(f"${c+1}$", (pos_x[c], pos_y[c]), annotation_clip=False, fontsize=13, va="center", ha="center")

    ax.add_patch(rectangle)
    fig.legend(loc="lower center", framealpha=1.0, fontsize=10, ncol=2)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.set_yticks([])

    # annotate inlet & outlet
    plt.arrow(-8.35, 1.75, 2, -1, color="black", head_width=0.25, clip_on=False)
    plt.arrow(17.35, 1.75, 2, -1, color="black", head_width=0.25, clip_on=False)
    plt.arrow(-6.75, -6.75, 0, 2, color="black", head_width=0.25, clip_on=False)
    plt.arrow(-6.75, -6.75, 2, 0, color="black", head_width=0.25, clip_on=False)

    plt.annotate("$inlet$", (-3 + xmin, h * 2 / 3 + 0.05 + ymin), annotation_clip=False, fontsize=12)
    plt.annotate("$outlet$", (-3.5 + l + xmin, h * 2 / 3 + 0.01 + ymin), annotation_clip=False, fontsize=12)
    plt.annotate("$\\frac{x}{d}$", (2.05 + xmin, -0.75 + ymin), annotation_clip=False, fontsize=16, va="center")
    plt.annotate("$\\frac{y}{d}$", (-1.5 + xmin, 0.25 + ymin), annotation_clip=False, fontsize=16)

    # annotate domain dimensions
    plt.annotate(f"${l / (2 * r[0])}$", (l / 2 + xmin, h + 1 + ymin), annotation_clip=False)
    plt.annotate(f"${h / (2 * r[0])}$", (l + 1 + xmin, h / 2 + ymin), annotation_clip=False)

    # annotate the dimensions & position of the domain
    pos = {"xy": [(xmin, ymin + h + 0.75), (xmin, h + ymin), (xmin + l, ymin + h), (xmin + l, ymin + h),
                  (xmin + l, ymin), (xmin + l + 0.75, ymin + h)],
           "xytxt": [(xmin + l, ymin + h + 0.75), (xmin, ymin + h + 1.25), (xmin + l, ymin + h + 1.25),
                     (xmin + l + 1.25, ymin + h), (xmin + l + 1.25, ymin), (xmin + l + 0.75, ymin)],
           "style": [("<->", "-"), ("-", "--"), ("-", "--"), ("-", "--"), ("-", "--"), ("<->", "-")]
           }
    for i in range(len(pos["style"])):
        plt.annotate("", xy=pos["xy"][i], xytext=pos["xytxt"][i], annotation_clip=False,
                     arrowprops=dict(arrowstyle=pos["style"][i][0], color="black", linestyle=pos["style"][i][1],
                                     mutation_scale=15))

    # annotate position of 1st cylinder
    plt.annotate("${:.2f}$".format((h - (pos_y[0] + r[0])) / (2 * r[0])), (pos_x[0] + 0.2, pos_y[0] + 6 * r[0]))
    plt.annotate("${:.2f}$".format(abs(pos_x[0] - r[0]) / (2 * r[0])), (pos_x[0] - 6 * r[0], pos_y[0] + 0.5 * r[0]))
    plt.annotate("", xy=(xmin, pos_y[0]), xytext=(pos_x[0] - r[0], pos_y[0]),
                 arrowprops=dict(arrowstyle="<->", color="black", linestyle="-", mutation_scale=15),
                 annotation_clip=False)
    plt.annotate("", xy=(pos_x[0], pos_y[0] + r[0]), xytext=(pos_x[0], ymin + h),
                 arrowprops=dict(arrowstyle="<->", color="black", linestyle="-", mutation_scale=15),
                 annotation_clip=False)

    ax.set_aspect("equal")
    fig.tight_layout()
    fig.subplots_adjust(left=0.1, right=0.92)
    plt.savefig(join("..", "plots", "rotatingPinball2D", "domain_pinball.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


if __name__ == "__main__":
    # path to the simulation
    path_cylinder = join("..", "data", "rotatingCylinder2D", "uncontrolled")
    path_pinball = join("..", "data", "rotatingPinball2D", "uncontrolled")

    # plot the domain for the cylinder2D simulation
    plot_numerical_setup_cylinder(path_cylinder)

    # plot the domain for the pinball simulation
    plot_numerical_setup_pinball(path_pinball)
