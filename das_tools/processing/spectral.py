import numpy as np
from scipy import signal


def bandlimited_windowed_rfft(x, fs, f_min, f_max):
    x = np.asarray(x)

    if x.ndim == 1:
        x = x[:, None]

    n = x.shape[0]

    window = signal.windows.hann(n, sym=False)[:, None]
    scale = np.sqrt(fs * np.sum(window**2))

    xw = x * window

    freq = np.fft.rfftfreq(n, d=1 / fs)
    X = np.fft.rfft(xw, axis=0) / scale

    keep = (freq >= f_min) & (freq <= f_max)

    return X[keep], freq[keep]

def bandlimited_rfft_freqs(fs, f_min, f_max, n):
    """
    Return the rFFT frequency bins between f_min and f_max.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    f_min : float
        Minimum frequency to keep in Hz.
    f_max : float
        Maximum frequency to keep in Hz.
    n : int
        Number of samples.

    Returns
    -------
    freq_keep : ndarray
        rFFT frequency bins in the requested band.
    """
    freq = np.fft.rfftfreq(n, d=1 / fs)
    keep = (freq >= f_min) & (freq <= f_max)
    return freq[keep]