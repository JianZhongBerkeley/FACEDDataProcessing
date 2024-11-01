import numpy as np

def vs_time_stamp_gen(
    static_moving_t_s,
    nof_orient,
    time_per_frame_ms,
):
    ms_to_s = 1e-3
    s_to_ms = 1e3

    orient_angles_deg = np.arange(nof_orient) * (360/nof_orient)
    orient_angles_rad = orient_angles_deg * (np.pi/180)

    nof_stim_steps = static_moving_t_s.size
    tot_trail_t_s = np.sum(static_moving_t_s) 

    stim_tstamp_s = np.zeros((nof_orient, nof_stim_steps, 2))

    cur_start_t_s = np.arange(nof_orient) * tot_trail_t_s
    for i_step in range(nof_stim_steps):
        stim_tstamp_s[:, i_step, 0] = cur_start_t_s
        cur_start_t_s = cur_start_t_s + static_moving_t_s[i_step]
        stim_tstamp_s[:, i_step, 1] = cur_start_t_s

    stim_tstamp = (stim_tstamp_s * s_to_ms)/time_per_frame_ms

    return (stim_tstamp, stim_tstamp_s, orient_angles_deg, orient_angles_rad)