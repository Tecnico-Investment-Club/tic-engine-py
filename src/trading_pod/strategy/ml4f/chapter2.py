
import math
import numpy as np

def entropy(x):
    """
    Computes the relative entropy of an indicator by binning values
    into equally spaced bins across the indicator's observed range.

    Parameters
    ----------
    x : array-like
        Indicator values.

    Returns
    -------
    float
        Relative entropy in the range [0, 1].
    """

    x = np.asarray(x, dtype=float)
    n = len(x)

    # Choose number of bins based on data size (exact rules from the book)
    if n >= 10000:
        nbins = 20
    elif n >= 1000:
        nbins = 10
    elif n >= 100:
        nbins = 5
    else:
        nbins = 3

    # Initialize bin counts
    counts = [0] * nbins

    # Find minimum and maximum values in the data
    xmin = float(x[0])
    xmax = float(x[0])
    for i in range(1, n):
        if x[i] < xmin:
            xmin = x[i]
        if x[i] > xmax:
            xmax = x[i]

    # Compute bin-scaling factor
    # Slightly reduce nbins to prevent out-of-range bin index
    # Add tiny constant to denominator to avoid division by zero
    factor = (nbins - 1e-11) / ( (xmax - xmin) + 1e-60 )

    # Count how many samples fall into each bin
    for i in range(n):
        k = int( factor * (x[i] - xmin) )  # bin index starting at 0
        counts[k] += 1

    # Compute entropy sum
    ent_sum = 0.0
    for c in counts:
        if c > 0:
            p = c / n
            ent_sum -= p * math.log(p)  # Equation (2.1)

    # Return relative entropy by dividing by log(K)
    return ent_sum / math.log(nbins)
