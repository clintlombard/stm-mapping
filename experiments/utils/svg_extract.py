import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy.interpolate import interp1d
from svgpathtools import svg2paths2


def extract_surface_from_svg(filename):
    paths, attributes, svg_attributes = svg2paths2(filename)
    path = paths[0]
    N = len(path)
    pts = np.zeros((N + 1, 2))
    for i, curve in enumerate(path):
        pts[i, 0] = curve.start.real
        pts[i, 1] = curve.start.imag

    pts[i + 1, 0] = curve.end.real
    pts[i + 1, 1] = curve.end.imag

    # Remove offset and scale
    pts -= pts[0, :]
    pts[:, 0] /= pts[-1, 0]
    maximum = np.max(pts[:, 1])
    pts[:, 1] /= maximum if maximum else 1

    f = interp1d(*pts.T)

    maxes = np.max(pts, axis=0)
    mins = np.min(pts, axis=0)
    x_lim = [mins[0], maxes[0]]
    y_lim = [mins[1], maxes[1]]
    return f, x_lim, y_lim


if __name__ == "__main__":
    f, x_lim, y_lim = extract_surface_from_svg("simonsberg.svg")
    ratio = np.sum(x_lim) / np.sum(y_lim)
    scale = 3
    fig = plt.figure(figsize=(ratio * scale, scale))

    x = np.random.rand(1000) * (x_lim[1] - x_lim[0]) + x_lim[0]
    y = f(x)

    plt.scatter(x, y)
    x = np.linspace(0, 1, 10000)
    y = f(x)
    plt.plot(x, y)

    plt.xlim(*x_lim)
    fig.tight_layout()
    plt.show()
