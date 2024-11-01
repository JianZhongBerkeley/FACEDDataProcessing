import numpy as np
import scipy.ndimage


def get_representative_ve_rois(src_data_cluster, snr_est_lp_sigma = 3):
    nof_slices = len(src_data_cluster)
    
    ve_slice_idxs = []
    ve_roi_idxs = []

    visited_roi = [None for _ in range(nof_slices)]
    for i_slice in range(nof_slices):
        cur_labeled_mask = src_data_cluster[i_slice]["src_labeled_mask"]
        cur_nof_rois = int(np.max(cur_labeled_mask))
        visited_roi[i_slice] = np.full((cur_nof_rois,), False)


    for i_slice in range(nof_slices):
        cur_labeled_mask = src_data_cluster[i_slice]["src_labeled_mask"]
        nof_rois = int(np.max(cur_labeled_mask))
        for i_roi in range(nof_rois):
            if visited_roi[i_slice][i_roi]:
                continue
            cur_forward_roi_idx = i_roi
            cur_forward_slice_idx = i_slice
            forward_linked_roi_idxs = []
            forward_linked_slice_idxs = []

            max_response = -np.inf
            max_response_slice_idx = -1
            max_response_roi_idx = -1
            while cur_forward_roi_idx >= 0:
                forward_linked_roi_idxs.append(cur_forward_roi_idx)
                forward_linked_slice_idxs.append(cur_forward_slice_idx)
                cur_roi_pass_t_test = src_data_cluster[cur_forward_slice_idx]["t_test_pass_mask"][cur_forward_roi_idx]
                cur_roi_pass_anonva_test = src_data_cluster[cur_forward_slice_idx]["anova_test_pass_mask"][cur_forward_roi_idx]
                if cur_roi_pass_t_test:
                    cur_trace = src_data_cluster[cur_forward_slice_idx]["FmFneu_continous_dFFs"][cur_forward_roi_idx,:]
                    cur_trace_lp = scipy.ndimage.gaussian_filter1d(cur_trace, sigma = snr_est_lp_sigma, axis = -1)
                    cur_trace_lp_pos = cur_trace_lp.copy()
                    cur_trace_lp_pos[cur_trace_lp_pos < 0] = 0
                    cur_snr = np.mean(cur_trace_lp_pos)/np.std(cur_trace - cur_trace_lp)
                    cur_max_response = cur_snr + 1E6 * cur_roi_pass_anonva_test
                    if cur_max_response > max_response:
                        max_response = cur_max_response
                        max_response_slice_idx = cur_forward_slice_idx
                        max_response_roi_idx = cur_forward_roi_idx
                    
                cur_forward_roi_pair_link = src_data_cluster[cur_forward_slice_idx]["forward_roi_pair_link"]
                if cur_forward_roi_pair_link is None:
                    cur_forward_roi_idx = -1
                    continue
                cur_forward_roi_idx = cur_forward_roi_pair_link[cur_forward_roi_idx]
                cur_forward_slice_idx = cur_forward_slice_idx + 1

            if max_response > 0:
                ve_slice_idxs.append(max_response_slice_idx)
                ve_roi_idxs.append(max_response_roi_idx)

            nof_linked_rois = len(forward_linked_roi_idxs)
                
            for i_roi_chain_idx in range(nof_linked_rois):
                cur_roi_idx = forward_linked_roi_idxs[i_roi_chain_idx]
                cur_slice_idx = forward_linked_slice_idxs[i_roi_chain_idx]
                visited_roi[cur_slice_idx][cur_roi_idx] = True

    ve_slice_idxs = np.array(ve_slice_idxs, dtype = int)
    ve_roi_idxs = np.array(ve_roi_idxs, dtype = int)

    return (ve_slice_idxs, ve_roi_idxs)



def get_representative_os_rois(src_data_cluster, snr_est_lp_sigma = 3):
    nof_slices = len(src_data_cluster)
    
    os_slice_idxs = []
    os_roi_idxs = []

    visited_roi = [None for _ in range(nof_slices)]
    for i_slice in range(nof_slices):
        cur_labeled_mask = src_data_cluster[i_slice]["src_labeled_mask"]
        cur_nof_rois = int(np.max(cur_labeled_mask))
        visited_roi[i_slice] = np.full((cur_nof_rois,), False)


    for i_slice in range(nof_slices):
        cur_labeled_mask = src_data_cluster[i_slice]["src_labeled_mask"]
        nof_rois = int(np.max(cur_labeled_mask))
        for i_roi in range(nof_rois):
            if visited_roi[i_slice][i_roi]:
                continue
            cur_forward_roi_idx = i_roi
            cur_forward_slice_idx = i_slice
            forward_linked_roi_idxs = []
            forward_linked_slice_idxs = []

            max_response = -np.inf
            max_response_slice_idx = -1
            max_response_roi_idx = -1
            while cur_forward_roi_idx >= 0:
                forward_linked_roi_idxs.append(cur_forward_roi_idx)
                forward_linked_slice_idxs.append(cur_forward_slice_idx)
                cur_roi_pass_t_test = src_data_cluster[cur_forward_slice_idx]["t_test_pass_mask"][cur_forward_roi_idx]
                cur_roi_pass_anonva_test = src_data_cluster[cur_forward_slice_idx]["anova_test_pass_mask"][cur_forward_roi_idx]
                if cur_roi_pass_t_test and cur_roi_pass_anonva_test:
                    cur_trace = src_data_cluster[cur_forward_slice_idx]["FmFneu_continous_dFFs"][cur_forward_roi_idx,:]
                    cur_trace_lp = scipy.ndimage.gaussian_filter1d(cur_trace, sigma = snr_est_lp_sigma, axis = -1)
                    cur_trace_lp_pos = cur_trace_lp.copy()
                    cur_trace_lp_pos[cur_trace_lp_pos < 0] = 0
                    cur_snr = np.mean(cur_trace_lp_pos)/np.std(cur_trace - cur_trace_lp)
                    cur_max_response = cur_snr
                    if cur_max_response > max_response:
                        max_response = cur_max_response
                        max_response_slice_idx = cur_forward_slice_idx
                        max_response_roi_idx = cur_forward_roi_idx
                    
                cur_forward_roi_pair_link = src_data_cluster[cur_forward_slice_idx]["forward_roi_pair_link"]
                if cur_forward_roi_pair_link is None:
                    cur_forward_roi_idx = -1
                    continue
                cur_forward_roi_idx = cur_forward_roi_pair_link[cur_forward_roi_idx]
                cur_forward_slice_idx = cur_forward_slice_idx + 1

            if max_response > 0:
                os_slice_idxs.append(max_response_slice_idx)
                os_roi_idxs.append(max_response_roi_idx)

            nof_linked_rois = len(forward_linked_roi_idxs)
                
            for i_roi_chain_idx in range(nof_linked_rois):
                cur_roi_idx = forward_linked_roi_idxs[i_roi_chain_idx]
                cur_slice_idx = forward_linked_slice_idxs[i_roi_chain_idx]
                visited_roi[cur_slice_idx][cur_roi_idx] = True

    os_slice_idxs = np.array(os_slice_idxs, dtype = int)
    os_roi_idxs = np.array(os_roi_idxs, dtype = int)

    return (os_slice_idxs, os_roi_idxs)


def get_representative_all_rois(src_data_cluster, snr_est_lp_sigma = 3):
    nof_slices = len(src_data_cluster)

    all_slice_idxs = []
    all_roi_idxs = []

    visited_roi = [None for _ in range(nof_slices)]
    for i_slice in range(nof_slices):
        cur_labeled_mask = src_data_cluster[i_slice]["src_labeled_mask"]
        cur_nof_rois = int(np.max(cur_labeled_mask))
        visited_roi[i_slice] = np.full((cur_nof_rois,), False)

    for i_slice in range(nof_slices):
        cur_labeled_mask = src_data_cluster[i_slice]["src_labeled_mask"]
        nof_rois = int(np.max(cur_labeled_mask))
        for i_roi in range(nof_rois):
            if visited_roi[i_slice][i_roi]:
                continue
            cur_forward_roi_idx = i_roi
            cur_forward_slice_idx = i_slice
            forward_linked_roi_idxs = []
            forward_linked_slice_idxs = []
            max_response = -np.inf
            max_response_slice_idx = -1
            max_response_roi_idx = -1
            while cur_forward_roi_idx >= 0:
                forward_linked_roi_idxs.append(cur_forward_roi_idx)
                forward_linked_slice_idxs.append(cur_forward_slice_idx)
                cur_roi_pass_t_test = src_data_cluster[cur_forward_slice_idx]["t_test_pass_mask"][cur_forward_roi_idx]
                cur_roi_pass_anonva_test = src_data_cluster[cur_forward_slice_idx]["anova_test_pass_mask"][cur_forward_roi_idx]
                cur_trace = src_data_cluster[cur_forward_slice_idx]["FmFneu_continous_dFFs"][cur_forward_roi_idx,:]
                cur_trace_lp = scipy.ndimage.gaussian_filter1d(cur_trace, sigma = snr_est_lp_sigma, axis = -1)
                cur_trace_lp_pos = cur_trace_lp.copy()
                cur_trace_lp_pos[cur_trace_lp_pos < 0] = 0
                cur_snr = np.mean(cur_trace_lp_pos)/np.std(cur_trace - cur_trace_lp)
                cur_max_response = cur_snr + 1E6 * cur_roi_pass_t_test + 1E8 * cur_roi_pass_t_test * cur_roi_pass_anonva_test
                if cur_max_response > max_response:
                    max_response = cur_max_response
                    max_response_slice_idx = cur_forward_slice_idx
                    max_response_roi_idx = cur_forward_roi_idx
                cur_forward_roi_pair_link = src_data_cluster[cur_forward_slice_idx]["forward_roi_pair_link"]
                if cur_forward_roi_pair_link is None:
                    cur_forward_roi_idx = -1
                    continue
                cur_forward_roi_idx = cur_forward_roi_pair_link[cur_forward_roi_idx]
                cur_forward_slice_idx = cur_forward_slice_idx + 1

            if max_response > 0:
                all_slice_idxs.append(max_response_slice_idx)
                all_roi_idxs.append(max_response_roi_idx)

            nof_linked_rois = len(forward_linked_roi_idxs)

            for i_roi_chain_idx in range(nof_linked_rois):
                cur_roi_idx = forward_linked_roi_idxs[i_roi_chain_idx]
                cur_slice_idx = forward_linked_slice_idxs[i_roi_chain_idx]
                visited_roi[cur_slice_idx][cur_roi_idx] = True

    all_slice_idxs = np.array(all_slice_idxs, dtype = int)
    all_roi_idxs = np.array(all_roi_idxs, dtype = int)

    return (all_slice_idxs, all_roi_idxs)
