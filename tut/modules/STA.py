import numpy as np

def time_and_current_vectors(T, d_t, V,interval=5e-3):
    """
    Generate the simulation time vector and the corresponding applied
    current vector.

    The function creates a uniformly spaced time vector from 0 to `T`
    with step size `d_t`. The applied current vector is constructed by
    assigning each value in `V` to a consecutive 5 ms interval.

    Parameters
    ----------
    T : float
        Total simulation time in seconds.
    d_t : float
        Simulation time step in seconds.
    V : array-like
        Sequence of current values. Each value is applied during a
        consecutive 5 ms interval.

    Returns
    -------
    time_vector : numpy.ndarray
        Array containing the simulation time values.
    applied_current_vector : numpy.ndarray
        Array containing the applied current value at each time step.

    Notes
    -----
    - The function assumes that `T` is a multiple of 5 ms.
    - The function assumes that `5e-3 / d_t` is an integer.
    """

    n_steps = int(round(T / d_t))
    time_vector = np.arange(n_steps) * d_t

    samples_per_interval = int(round(interval / d_t))

    applied_current_vector = np.repeat(V, samples_per_interval)

    # Safety check
    if len(applied_current_vector) != len(time_vector):
        raise ValueError(
            f"Length mismatch: "
            f"time_vector={len(time_vector)}, "
            f"applied_current_vector={len(applied_current_vector)}"
        )

    return time_vector, applied_current_vector

def expandbin(old_vector, old_dt, new_dt):
    """
    Downsample a time series by averaging consecutive bins.

    This function groups the input vector into blocks of size
    `new_dt / old_dt` and replaces each block with its average value.

    Parameters
    ----------
    old_vector : array-like
        Original signal or time series sampled with timestep `old_dt`.

    old_dt : float
        Original sampling interval.

    new_dt : float
        Desired sampling interval. Must be an integer multiple
        of `old_dt`.

    Returns
    -------
    numpy.ndarray
        Downsampled vector with timestep `new_dt`, where each
        element is the average of the corresponding block from
        `old_vector`.

    Notes
    -----
    If `r = new_dt / old_dt`, then:
        len(new_vector) = len(old_vector) / r

    The function assumes:
    - `new_dt >= old_dt`
    - `new_dt / old_dt` is an integer
    - `len(old_vector)` is divisible by `r`
    """
    
    old_l = len(old_vector)

    r = int(new_dt / old_dt)

    new_l = old_l / r

    new_vector = np.zeros(int(new_l))

    for i in range(int(new_l)):

        for j in range(r):

            new_vector[i] += old_vector[i * r + j] / r

    return new_vector

def downsample_spikes(spikes, old_dt, new_dt):
    """
    Downsample a spike train by averaging adjacent bins and converting
    the result back into a binary spike representation.

    Parameters
    ----------
    spikes : array-like
        Binary spike train sampled with timestep `old_dt`.
        Typically contains 0s and 1s.
    old_dt : float
        Original timestep of the spike train.
    new_dt : float
        Desired timestep after downsampling.
        Must be an integer multiple of `old_dt`.

    Returns
    -------
    numpy.ndarray
        Downsampled binary spike train. Each output bin is set to 1
        if at least one spike occurred within the corresponding
        aggregation window; otherwise it is 0.

    Notes
    -----
    This function first applies `expandbin()` to average groups of
    adjacent bins. Since averaging may produce fractional values,
    all nonzero entries are then thresholded back to 1.
    """
    
    new_spikes = expandbin(spikes, old_dt, new_dt)

    for i in range(len(new_spikes)):

        if new_spikes[i] > 0:

            new_spikes[i] = 1

    return new_spikes

def STA(applied_current_vector, spikes, dt, t_minus=75, t_plus=25):
    """
    Compute the spike-triggered average (STA) of an input current signal.

    The STA is obtained by averaging segments of the applied current
    surrounding each spike occurrence. For every spike, a temporal
    window extending `t_minus` before the spike and `t_plus` after
    the spike is extracted and accumulated. The final STA is the mean
    of all valid spike-centered windows.

    Parameters
    ----------
    applied_current_vector : array-like
        Input current time series.

    spikes : array-like
        Binary spike train with the same length as
        `applied_current_vector`. Spike occurrences must be marked
        with value 1.

    dt : float
        Simulation time step.

    t_minus : float, optional
        Time interval before each spike to include in the STA window.
        Default is 75.

    t_plus : float, optional
        Time interval after each spike to include in the STA window.
        Default is 25.

    Returns
    -------
    sta : numpy.ndarray
        Spike-triggered average current waveform.

    time_window : numpy.ndarray
        Time axis corresponding to the STA window, ranging from
        `-t_minus` to `t_plus`.

    Raises
    ------
    ValueError
        If `applied_current_vector` and `spikes` do not have the
        same length.

    Notes
    -----
    Spike windows that would exceed the signal boundaries are ignored.
    """
    
    if len(applied_current_vector) != len(spikes):
        raise ValueError("applied_current_vector and spikes must have the same length.")
    
    n_minus = int(t_minus/dt)
    n_plus = int(t_plus/dt)

    time_window = np.arange(-n_minus*dt, n_plus*dt+dt, dt)

    sta = np.zeros(len(time_window))

    # find time bins where spikes occur:
    spike_bins = np.where(spikes == 1)[0]

    for spike_bin in spike_bins:

        start = spike_bin - n_minus
        end = spike_bin + n_plus

        if start < 0 or end >= len(applied_current_vector):
            continue

        sta += applied_current_vector[start:end+1]

    sta /= len(spike_bins)

    return sta, time_window