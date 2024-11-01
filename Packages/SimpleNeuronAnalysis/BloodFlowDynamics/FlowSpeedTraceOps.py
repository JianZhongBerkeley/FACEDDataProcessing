import numpy as np
import scipy.ndimage
import scipy.signal
import pykalman 


def stats_outlier_removal_valid_mask(src_trace, std_ratio = 3, nof_itr = 3):
    assert len(src_trace.shape) == 1

    valid_mask = np.full(src_trace.shape, True)
    for i_itr in range(nof_itr):
        cur_mean = np.mean(src_trace[valid_mask])
        cur_std = np.std(src_trace[valid_mask])
        cur_valid_mask = np.abs(src_trace - cur_mean) < std_ratio*cur_std
        valid_mask = np.logical_and(valid_mask, cur_valid_mask)

    return valid_mask


def stats_outlier_removal(src_trace, std_ratio = 3, nof_itr = 3):
    assert len(src_trace.shape) == 1

    valid_mask = stats_outlier_removal_valid_mask(src_trace, std_ratio, nof_itr)
    dst_trace = src_trace.copy()
    dst_trace[~valid_mask] = np.mean(dst_trace[valid_mask])
    return dst_trace


def win_stats_outlier_removal(src_trace, win_size = 5, std_ratio = 3, nof_itr = 3, pad_mode = "wrap"):
    assert len(src_trace.shape) == 1

    win_size = 2*(win_size//2) + 1
    
    trace_len = len(src_trace)
    dst_trace = src_trace.copy()
    
    for i_itr in range(nof_itr):
        pad_trace = np.pad(dst_trace, (win_size//2, win_size//2), mode = pad_mode).copy()
        for i_win_idx in range(trace_len):
            win_start_idx = i_win_idx
            win_end_idx = i_win_idx + win_size 
            win_trace = pad_trace[win_start_idx:win_end_idx]
            win_mean = np.nanmean(win_trace)
            win_std = np.nanstd(win_trace)
            if np.abs(dst_trace[i_win_idx] - win_mean) > std_ratio*win_std:
                dst_trace[i_win_idx] = win_mean
        
    return dst_trace


def trace_kalman_filter(src_trace, observ_covar_amp = 1E3, filter_mod = 0):
    assert len(src_trace.shape) == 1

    kalman_measurements = src_trace.reshape((-1,1))
    kalman_init_state_mean = np.array([src_trace[0], 0])
    kalman_transition_matrix = np.array([[1, 1],
                                        [0, 1]])
    kalman_observ_matrix = np.array([[1, 0]])
    
    kalman_filter = pykalman.KalmanFilter(transition_matrices  = kalman_transition_matrix,
                                         observation_matrices = kalman_observ_matrix,
                                         initial_state_mean = kalman_init_state_mean)
    kalman_filter = kalman_filter.em(kalman_measurements, n_iter = 5)
    
    kalman_filter = pykalman.KalmanFilter(transition_matrices  = kalman_transition_matrix,
                                         observation_matrices = kalman_observ_matrix,
                                         initial_state_mean = kalman_init_state_mean,
                                         observation_covariance = observ_covar_amp*kalman_filter.observation_covariance,
                                         em_vars=['transition_covariance', 'initial_state_covariance'])
    kalman_filter = kalman_filter.em(kalman_measurements, n_iter = 5)
    
    smoothed_state_means = None
    smoothed_state_covariances = None
    if filter_mod == 1:
        (smoothed_state_means, smoothed_state_covariances) = kalman_filter.filter(kalman_measurements)
    else:
        (smoothed_state_means, smoothed_state_covariances) = kalman_filter.smooth(kalman_measurements)

    dst_trace = np.matmul(smoothed_state_means, np.transpose(kalman_observ_matrix))

    dst_trace = np.squeeze(dst_trace)
    
    return dst_trace


def eva_trace(src_trace):
    low_pass_trace = scipy.ndimage.gaussian_filter1d(src_trace, sigma = 5)
    return np.abs(np.mean(low_pass_trace)/np.mean(np.abs(src_trace - low_pass_trace)))