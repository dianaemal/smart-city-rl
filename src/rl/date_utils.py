import numpy as np


def split_date_indices(num_dates, episode_length, forecast_horizon=0, holdout_fraction=0.15):
    """
    Splits valid episode start-indices into a TRAIN pool and a held-out
    EVAL pool, leaving a gap of (episode_length + forecast_horizon - 1)
    between them so that no training episode's forecast window can run
    forward into eval territory (and vice versa).

    The eval pool is the most recent contiguous block of dates - this
    mirrors how the model would actually be used (trained on the past,
    evaluated on the most recent/unseen period).
    """
    span = episode_length + max(forecast_horizon - 1, 0)
    max_start = num_dates - span               # last valid start index
    total_starts = max_start + 1
    n_holdout = int(total_starts * holdout_fraction)

    eval_starts = np.arange(total_starts - n_holdout, total_starts)
    boundary = eval_starts[0]

    train_end = max(boundary - span, 0)
    train_starts = np.arange(0, train_end + 1)

    return train_starts, eval_starts