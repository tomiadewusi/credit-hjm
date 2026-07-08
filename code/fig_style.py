# fig_style.py
# This code was created with the assistance of ChatCPT. 
# I wanted to get IEEE style graphs and it helped me get
# everything in the correct format

import matplotlib as mpl
import matplotlib.pyplot as plt

# Rough IEEE column widths in inches
_SINGLE_COL_WIDTH = 3.5      # about 8.9 cm
_DOUBLE_COL_WIDTH = 7.16     # about 18.2 cm


def use_ieee_style(single_column=True, height_factor=1.0, use_tex=True):
    """
    Set global Matplotlib rcParams to something close to IEEE style
    in grayscale.

    Parameters
    ----------
    single_column : bool
        If True, use single column width. If False, use double column width.
    height_factor : float
        Height as a multiple of the figure width. For example 1.0 or 0.75.
    use_tex : bool
        If True, enable LaTeX text rendering (requires a working LaTeX install).
    """
    width = _SINGLE_COL_WIDTH if single_column else _DOUBLE_COL_WIDTH
    height = width * height_factor

    rc = {
        # Figure
        "figure.figsize": (width, height),
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.format": "pdf",

        # Fonts
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,

        # Grid
        "axes.grid": False,
        "grid.alpha": 0.3,

        # Grayscale cycle
        "axes.prop_cycle": mpl.cycler("color", ["black"]),

        # Lines
        "lines.linewidth": 1.0,
    }

    if use_tex:
        rc["text.usetex"] = True
        rc["font.family"] = "serif"

    mpl.rcParams.update(rc)


def new_figure(nrows=1, ncols=1, single_column=True, height_factor=1.0,
               sharex=False, sharey=False):
    """
    Create a new IEEE style figure with a consistent size.

    Parameters
    ----------
    nrows, ncols : int
        Subplot grid layout.
    single_column : bool
        Single or double column width.
    height_factor : float
        Height as a multiple of width.
    sharex, sharey : bool
        Matplotlib sharex and sharey flags.

    Returns
    -------
    fig, axes
    """
    width = _SINGLE_COL_WIDTH if single_column else _DOUBLE_COL_WIDTH
    height = width * height_factor

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(width, height),
        sharex=sharex,
        sharey=sharey,
    )
    return fig, axes


def save_figure(fig, filename, tight=True):
    """
    Save a figure as vector PDF with tight bounding box.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    filename : str
        Output file name, for example "credit-spreads-bps.pdf".
    tight : bool
        If True, apply tight_layout before saving.
    """
    if tight:
        fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
