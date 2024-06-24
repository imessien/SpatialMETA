import numpy as np
from scipy.stats import zscore
from scipy.stats import linregress

# Normalize numeric values
def normalize_func(x):
    x = np.array(x)
    return (x - np.min(x)) / (np.max(x) - np.min(x))

# Zscore computation
def normalize_zscore(x):
    return zscore(x)

def two_peaks(arr):
    return np.sin(np.linspace(1.5 * np.pi, 5.5 * np.pi, len(arr)))

def linear(arr):
    return np.linspace(0, 1, len(arr))

def sinus(arr):
    return np.sin(np.linspace(0, 2 * np.pi, len(arr)))

def gradient(arr):
    return np.cos(np.linspace(0, np.pi, len(arr)))

def log(arr):
    return np.log(np.arange(1, len(arr) + 1))

def early_peak(arr):
    half_length = len(arr) // 2
    prel_res = np.sin(np.linspace(1.5 * np.pi, 3.5 * np.pi, half_length))
    res = np.concatenate((prel_res, np.repeat(np.min(prel_res), half_length)))
    return res

def late_peak(arr):
    half_length = len(arr) // 2
    prel_res = np.sin(np.linspace(1.5 * np.pi, 3.5 * np.pi, half_length))
    res = np.concatenate((np.repeat(np.min(prel_res), half_length), prel_res))
    return res

def abrupt_ascending(arr):
    min_arr = np.min(arr)
    max_arr = np.max(arr)
    len_arr = len(arr)
    len_by_part = len_arr // 3
    seq1 = np.arange(1, len_by_part + 1)
    seq3 = np.arange(len_arr - len_by_part, len_arr + 1)
    seq2 = np.arange(np.max(seq1) + 1, np.min(seq3))
    curve1 = np.repeat(min_arr, len(seq1))
    curve3 = np.repeat(max_arr, len(seq3))
    curve2 = np.flip(gradient(np.linspace(min_arr, max_arr, len(seq2))))
    out = np.concatenate((curve1, curve2, curve3))
    return out

def abrupt_descending(arr):
    return np.flip(abrupt_ascending(arr))

def immediate_ascending(arr):
    min_arr = np.min(arr)
    max_arr = np.max(arr)
    len_arr = len(arr)
    len_by_part = len_arr // 5
    seq1 = np.arange(1, len_by_part + 1)
    seq2 = np.arange(len_by_part + 1, 2 * len_by_part + 1)
    seq4 = np.arange(3 * len_by_part + 1, 4 * len_by_part + 1)
    seq5 = np.arange(4 * len_by_part + 1, 5 * len_by_part + 1)
    seq3 = np.arange(np.max(seq2) + 1, np.min(seq4))
    curve1 = np.repeat(min_arr, len(seq1))
    curve2 = np.repeat(min_arr, len(seq2))
    curve4 = np.repeat(max_arr, len(seq4))
    curve5 = np.repeat(max_arr, len(seq5))
    curve3 = np.flip(gradient(np.linspace(min_arr, max_arr, len(seq3))))
    out = np.concatenate((curve1, curve2, curve3, curve4, curve5))
    return out

def immediate_descending(arr):
    return np.flip(immediate_ascending(arr))

def one_peak(arr):
    return np.sin(np.linspace(1.5 * np.pi, 3.5 * np.pi, len(arr)))

def sharp_peak(arr):
    min_arr = np.min(arr)
    max_arr = np.max(arr)
    len_arr = len(arr)
    len_by_part = len_arr // 3
    seq1 = np.arange(1, len_by_part + 1)
    seq3 = np.arange(len_arr - len_by_part, len_arr + 1)
    seq2 = np.linspace(np.max(seq1) + 1, np.min(seq3) - 1, len(seq1) + len(seq3) - 2)
    curve1 = np.repeat(min_arr, len(seq1))
    curve3 = np.repeat(min_arr, len(seq3))
    curve2 = one_peak(seq2)
    out = np.concatenate((curve1, curve2, curve3))
    return out

# List of valid curve functions
valid_curves = {
    "early_peak": early_peak,
    "gradient": gradient,
    "late_peak": late_peak,
    "linear": linear,
    "log": log,
    "one_peak": one_peak,
    "sinus": sinus,
    "two_peaks": two_peaks,
    "abrupt_ascending": abrupt_ascending,
    "abrupt_descending": abrupt_descending,
    "immediate_ascending": immediate_ascending,
    "immediate_descending": immediate_descending,
    "sharp_peak": sharp_peak
}

# Curve fitting
def fit_curve(arr, fn, rev=False, normalize=True):
    assert(fn in valid_curves.keys())

    out = valid_curves[fn](arr)

    # Add conditions for other curve functions

    if rev == 1 or rev == "x":
        out = np.flip(out)

    elif rev == 2 or rev == "y":
        mini = np.min(arr)
        maxi = np.max(arr)
        out *= -1
        out = (out - np.min(out)) * (maxi - mini) / (np.max(out) - np.min(out)) + mini

    if normalize:
        out = normalize_func(out)

    return out

def fit_p_value(input, fn):
    # Fit the curve
    fitted_curve = fit_curve(input, fn)
    
    # Perform linear regression to calculate R-squared
    slope, intercept, r_value, p_value, std_err = linregress(input, fitted_curve)
    
    return p_value