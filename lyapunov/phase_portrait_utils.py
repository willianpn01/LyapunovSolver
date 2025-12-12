import numpy as np


def build_stream_seeds(
    seed_mode: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    grid_n: int = 12,
    border_n: int = 25,
    margin_frac: float = 0.02,
):
    if seed_mode == "Automático":
        return None

    xr = float(x_max - x_min)
    yr = float(y_max - y_min)
    xm = float(x_min + margin_frac * xr)
    xM = float(x_max - margin_frac * xr)
    ym = float(y_min + margin_frac * yr)
    yM = float(y_max - margin_frac * yr)

    if seed_mode == "Grade (controlado)":
        grid_n = int(grid_n)
        xs = np.linspace(xm, xM, grid_n)
        ys = np.linspace(ym, yM, grid_n)
        XXs, YYs = np.meshgrid(xs, ys)
        return np.column_stack([XXs.ravel(), YYs.ravel()])

    if seed_mode == "Borda (controlado)":
        border_n = int(border_n)
        xs = np.linspace(xm, xM, border_n)
        ys = np.linspace(ym, yM, border_n)
        top = np.column_stack([xs, np.full_like(xs, yM)])
        bottom = np.column_stack([xs, np.full_like(xs, ym)])
        left = np.column_stack([np.full_like(ys, xm), ys])
        right = np.column_stack([np.full_like(ys, xM), ys])
        return np.vstack([top, bottom, left, right])

    raise ValueError(f"Unsupported seed_mode: {seed_mode}")
