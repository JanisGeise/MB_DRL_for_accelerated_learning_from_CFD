"""
    plot the numerical setup of the cylinder2D and pinball
"""
import regex as re
import torch as pt

from os.path import join
from os import path, makedirs
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc, RegularPolygon
from numpy import radians as rad
import numpy as np


def get_probe_locations(load_dir: str) -> pt.Tensor:
    pattern = r"-?\d.\d+ -?\d.\d+ -?\d.\d+"
    with open(join(load_dir, "system", "controlDict"), "r") as f:
        loc = f.readlines()

    # avoid finding other coordinate tuples, which may be present in the controlDict
    idx = [i for i, l in enumerate(loc) if "probeLocations" in l][0]

    # get coordinates of probes, omit appending empty lists and map strings to floats
    coord = [re.findall(pattern, line) for i, line in enumerate(loc) if re.findall(pattern, line) and i > idx]
    return pt.tensor([list(map(float, i[0].split())) for i in coord])


def drawCirc(ax, radius, centX, centY, angle_, theta2_, color_='black'):
    """Draw partial circle with arrow.

    Taken and modified from:
    https://stackoverflow.com/questions/37512502/how-to-make-arrow-that-loops-in-matplotlib
    """
    arc = Arc([centX,centY],radius,radius,angle=angle_,
          theta1=0, theta2=theta2_, capstyle='round', linestyle='-', lw=1, color=color_)
    ax.add_patch(arc)
    endX=centX+(radius/2)*np.cos(rad(theta2_+angle_))
    endY=centY+(radius/2)*np.sin(rad(theta2_+angle_))

    ax.add_patch(                    #Create triangle as arrow head
        RegularPolygon(
            (endX, endY),            # (x,y)
            3,                       # number of vertices
            radius/9,                # radius
            rad(angle_+theta2_),     # orientation
            color=color_
        )
    )
    #ax.set_xlim([centX-radius,centY+radius]) and ax.set_ylim([centY-radius,centY+radius]) 


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
    l *= 0.5

    # plot cylinder, probe locations and annotate the domain
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.scatter(pos_probes[:, 0], pos_probes[:, 1], marker="x",  c="C1", s=10, linewidth=1)

    # cylinders and frame
    circle = Circle((pos_x, pos_y), radius=r, color="gray")
    rectangle = Rectangle((0, 0), width=l, height=h, edgecolor="black", linewidth=2, facecolor="none")
    ax.add_patch(circle)
    ax.add_patch(rectangle)
    ax.set_xlim(0, l)
    ax.set_ylim(0, h)
    ax.set_xticks([])
    ax.set_yticks([])

    # coordinate system
    ax.arrow(0, 0, 0.1, 0.0, color="C3", lw=1.5, head_width=0.02, clip_on=False, zorder=5)
    ax.arrow(0, 0, 0.0, 0.1, color="C3", lw=1.5, head_width=0.02, clip_on=False, zorder=5)
    ax.text(0.1, -0.04, r"$x$")
    ax.text(-0.04, 0.1, r"$y$")

    # annotate inlet, outlet, and walls
    ax.text(-0.04, h*0.5, "inlet", rotation=90, va="center")
    ax.text(l-0.04, h*0.5, "outlet", rotation=90, va="center")
    ax.text(0.5*l, 0.01, "wall")
    ax.text(0.5*l, h-0.035, "wall")

    # annotate domain dimensions
    anno_props = {
        "arrowprops" : dict(arrowstyle='<->', shrinkA=0, shrinkB=0),
        "color" : "k",
        "annotation_clip" : False
    }
    ax.annotate("", xy=(0, h+0.02), xytext=(l, h+0.02), **anno_props)
    ax.annotate("", xy=(l+0.02, 0), xytext=(l+0.02, h), **anno_props)
    ax.annotate("", xy=(0, 0.2), xytext=(0.15, 0.2), **anno_props)
    ax.annotate("", xy=(0.2, 0.25), xytext=(0.2, 0.41), **anno_props)
    ax.annotate("", xy=(0.2-0.05, 0.1), xytext=(0.2+0.05, 0.1), **anno_props)
    anno_props = {
        "arrowprops" : dict(arrowstyle='-', shrinkA=0, shrinkB=0, linestyle="--"),
        "color" : "k",
        "annotation_clip" : False
    }
    ax.annotate("", xy=(l, 0), xytext=(l+0.03, 0), **anno_props)
    ax.annotate("", xy=(l, h), xytext=(l+0.03, h), **anno_props)
    ax.annotate("", xy=(0, h), xytext=(0, h+0.03), **anno_props)
    ax.annotate("", xy=(l, h), xytext=(l, h+0.03), **anno_props)
    ax.annotate("", xy=(0.15, 0.1-0.01), xytext=(0.15, 0.2), **anno_props)
    ax.annotate("", xy=(0.25, 0.1-0.01), xytext=(0.25, 0.2), **anno_props)
    ax.text(0.5*l, h+0.03, r"$22d$", ha="center")
    ax.text(0.095, 0.2+0.01, r"$1.5d$", ha="center")
    ax.text(0.2, 0.1+0.01, r"$d$", ha="center")
    ax.text(l+0.03, 0.5*h, r"$4.1d$", rotation=90, va="center")
    ax.text(0.2-0.035, 0.325, r"$1.6d$", rotation=90, va="center")
    ax.scatter(pos_x, pos_y, marker="+", c="k", s=15, linewidth=1)

    drawCirc(ax, 0.04, 0.2, 0.2, 0, 240, "C1")
    ax.text(0.2+0.02, 0.2-0.02, r"$\omega$", c="C1", va="center", ha="center")

    # velocity profile
    y = pt.linspace(0.005, h-0.005, 40)
    vel = (1 - 4*(0.5-y/h)**2)*0.06
    ax.plot(vel, y, c="C0")
    y = pt.linspace(0, h, 10)
    vel = (1 - 4*(0.5-y/h)**2)*0.06
    anno_props = {
        "arrowprops" : dict(arrowstyle='<-', shrinkA=0, shrinkB=1.5, linestyle="-", color="C0"),
        "color" : "C0",
        "annotation_clip" : False
    }
    for (yi, vi) in zip(y, vel):
        ax.annotate("", xy=(0, yi.item()), xytext=(vi.item(), yi.item()),**anno_props)

    ax.set_aspect("equal")
    plt.savefig(join("..", "plots", "rotatingCylinder2D", "domain_cylinder.pdf"), bbox_inches="tight")
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
    norm = 6
    xmax /= norm * 1.5
    xmin /= norm
    ymax /= norm
    ymin /= norm

    l = xmax - xmin
    h = ymax - ymin

    # get cylinder positions
    norm_in = 3
    pos_x = [float(loc[i].strip(";\n").split()[1]) / norm_in for i in range(76, 79)]
    pos_y = [float(loc[i].strip(";\n").split()[1]) / norm_in for i in range(79, 82)]
    r = [float(loc[i].strip(";\n").split()[1]) / norm_in for i in range(82, 85)]

    # probe locations
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.scatter(pos_probes[:, 0]/norm_in, pos_probes[:, 1]/norm_in, marker="x", color="C1", s=10, linewidth=1, zorder=6)

    # symmetry lines
    ax.plot([xmin, xmax], [0, 0], ls="--", lw=1, c="k")
    ax.plot([0, 0], [ymin, ymax], ls="--", lw=1, c="k")

    # plot cylinders
    for c in range(len(r)):
        circle = Circle((pos_x[c], pos_y[c]), radius=r[c], color="gray")
        ax.add_patch(circle)
        ax.scatter(pos_x[c], pos_y[c], marker="+", c="k", s=15, linewidth=1)
        drawCirc(ax, 0.2, pos_x[c], pos_y[c], 0, 240, "C1")
        ax.text(pos_x[c]+0.05, pos_y[c]-0.05, rf"$\omega_{c+1}$", c="C1", va="center", ha="center")

    # inlet velocity
    ax.plot([xmin+0.125, xmin+0.125, xmin+0.1, xmin+0.1], [ymin, 0, 0, ymax], c="C0", lw=1)
    y = pt.linspace(ymin*0.9, ymax*0.9, 10)
    vel = [xmin+0.125 if yi.item() < 0 else xmin+0.1 for yi in y]
    anno_props = {
        "arrowprops" : dict(arrowstyle='<-', shrinkA=0, shrinkB=1, linestyle="-", color="C0", mutation_scale=8),
        "color" : "C0",
        "annotation_clip" : False
    }
    for (yi, vi) in zip(y, vel):
        plt.annotate("", xy=(xmin, yi.item()), xytext=(vi, yi.item()),**anno_props)
    ax.text(xmin+0.135, 0.5*ymin, r"$U_\mathrm{in}+\varepsilon$", rotation=-90, va="center", c="C0")
    ax.text(xmin+0.12, 0.5*ymax, r"$U_\mathrm{in}-\varepsilon$", rotation=-90, va="center", c="C0")


    # frame
    rectangle = Rectangle((xmin, ymin), width=l, height=h, edgecolor="black", linewidth=2, facecolor="none")
    ax.add_patch(rectangle)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.set_yticks([])

    # coordinate system
    ax.arrow(0, 0, 0.1, 0.0, color="C3", lw=1.5, head_width=0.02, clip_on=False, zorder=5)
    ax.arrow(0, 0, 0.0, 0.1, color="C3", lw=1.5, head_width=0.02, clip_on=False, zorder=5)
    ax.text(0.1, -0.09, r"$x$")
    ax.text(-0.09, 0.075, r"$y$")

    # domain dimensions
    anno_props = {
        "arrowprops" : dict(arrowstyle='<->', shrinkA=0, shrinkB=0),
        "color" : "k",
        "annotation_clip" : False
    }
    ax.annotate("", xy=(xmin, ymax+0.05), xytext=(xmax, ymax+0.05), **anno_props)
    ax.annotate("", xy=(0.75*xmax, 0), xytext=(0.75*xmax, ymax), **anno_props)
    ax.annotate("", xy=(0.75*xmax, 0), xytext=(0.75*xmax, ymin), **anno_props)
    ax.annotate("", xy=(xmin, 0.75*ymin), xytext=(0, 0.75*ymin), **anno_props)
    ax.annotate("", xy=(xmin*0.65, -0.5/norm_in), xytext=(xmin*0.65, 0.5/norm_in), **anno_props)
    ax.annotate("", xy=(1/norm_in, -0.75/norm_in), xytext=(1/norm_in, 0.75/norm_in), **anno_props)
    ax.annotate("", xy=(-1.3/norm_in, 0.5*ymax), xytext=(0, 0.5*ymax), **anno_props)
    anno_props = {
        "arrowprops" : dict(arrowstyle='-', shrinkA=0, shrinkB=0, linestyle="--"),
        "color" : "k",
        "annotation_clip" : False
    }
    ax.annotate("", xy=(xmin, ymax), xytext=(xmin, ymax+0.075), **anno_props)
    ax.annotate("", xy=(xmax, ymax), xytext=(xmax, ymax+0.075), **anno_props)
    ax.annotate("", xy=(xmin*0.65-0.025, 0.5/norm_in), xytext=(-1.3/norm_in, 0.5/norm_in), **anno_props)
    ax.annotate("", xy=(xmin*0.65-0.025, -0.5/norm_in), xytext=(-1.3/norm_in, -0.5/norm_in), **anno_props)
    ax.annotate("", xy=(0, 0.75/norm_in), xytext=(1/norm_in+0.025, 0.75/norm_in), **anno_props)
    ax.annotate("", xy=(0, -0.75/norm_in), xytext=(1/norm_in+0.025, -0.75/norm_in), **anno_props)
    ax.annotate("", xy=(-1.3/norm_in, 0), xytext=(-1.3/norm_in, 0.5*ymax+0.025), **anno_props)
    ax.text(xmin*0.65-0.1, 0.05, r"$d$", va="center", rotation=90)
    ax.text(1/norm_in-0.1, 0.05, r"$1.5d$", va="center", rotation=90)
    ax.text(0.75*xmax-0.1, 0.5*ymax, r"$6d$", va="center", rotation=90)
    ax.text(0.75*xmax-0.1, 0.5*ymin, r"$6d$", va="center", rotation=90)
    ax.text(0.5*(xmin+xmax), ymax+0.075, r"$22d$", ha="center")
    ax.text(-0.65/norm_in, 0.5*ymax+0.025, r"$1.3d$", ha="center")
    ax.text(0.5*xmin, 0.75*ymin+0.025, r"$6d$", ha="center")

    ax.text(xmin-0.125, 0, "inlet", rotation=90, va="center")
    ax.text(xmax-0.125, 1/norm_in, "outlet", rotation=90, va="center")

    ax.set_aspect("equal")
    plt.savefig(join("..", "plots", "rotatingPinball2D", "domain_pinball.pdf"), bbox_inches="tight")
    plt.close("all")


if __name__ == "__main__":
    # path to the simulation
    path_cylinder = join("..", "data", "rotatingCylinder2D", "uncontrolled")
    path_pinball = join("..", "data", "rotatingPinball2D", "uncontrolled")

    # plot the domain for the cylinder2D simulation
    plot_numerical_setup_cylinder(path_cylinder)

    # plot the domain for the pinball simulation
    plot_numerical_setup_pinball(path_pinball)
