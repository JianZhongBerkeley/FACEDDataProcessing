import numpy as np


def norm_cov_matrix(m):
    assert len(m.shape) == 2
    assert m.shape[0] == m.shape[1]
        
    norm_m = np.zeros(m.shape)
    for i_elem in range(m.shape[0]):
        for j_elem in range(m.shape[1]):
            norm_m[i_elem,j_elem] = m[i_elem,j_elem]/np.sqrt(m[i_elem,i_elem]*m[j_elem,j_elem])
    return norm_m


def shift_correct_cov_matrix(src_ns):
    assert len(src_ns.shape) == 3
    roll_shift = -1
    
    nof_rois, nof_trials, trace_len = src_ns.shape
    roll_ns = np.roll(src_ns, roll_shift, axis = 1) 

    cov_ns = np.zeros((nof_rois,nof_rois))
    for i_roi in range(nof_rois):
        for j_roi in range(i_roi, nof_rois):
            cur_nij_prod = np.mean(src_ns[i_roi,:,:]*src_ns[j_roi,:,:], axis = -1)
            cur_roll_nij_prod = np.mean(src_ns[i_roi,:,:]*roll_ns[j_roi,:,:], axis = -1)
            cov_ns[i_roi, j_roi] = np.mean(cur_nij_prod - cur_roll_nij_prod)
            cov_ns[j_roi, i_roi] = cov_ns[i_roi, j_roi]
    return cov_ns
    