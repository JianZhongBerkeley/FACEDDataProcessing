import numpy as np
from .Utils import mix_dFF_group

def add_roi_center_ijs(src_data_cluster):
    nof_slices = len(src_data_cluster)
    for i_slice in range(nof_slices):
        cur_labeled_mask = src_data_cluster[i_slice]["src_labeled_mask"]
        cur_roi_idx_to_label_offset = src_data_cluster[i_slice]["roi_idx_to_label_offset"]
        cur_slice_num = src_data_cluster[i_slice]["slice_num"]
        nof_rois = int(np.max(cur_labeled_mask))
        cur_roi_center_ijs = np.zeros((nof_rois,2))
        cur_roi_center_yxz_ums = np.zeros((nof_rois,3))
        for i_roi in range(nof_rois):
            roi_label = i_roi + cur_roi_idx_to_label_offset
            roi_mask = (cur_labeled_mask == roi_label)
            roi_mask_is, roi_mask_js = np.where(roi_mask)
            roi_mask_center_i = np.mean(roi_mask_is)
            roi_mask_center_j = np.mean(roi_mask_js)
            cur_roi_center_ijs[i_roi, 0] = roi_mask_center_i
            cur_roi_center_ijs[i_roi, 1] = roi_mask_center_j
        src_data_cluster[i_slice]["roi_center_ijs"] = cur_roi_center_ijs


def add_roi_center_yxz_ums(src_data_cluster, image_ij_pixelsize_um, image_z_stepsize_um):
    nof_slices = len(src_data_cluster)
    for i_slice in range(nof_slices):
        cur_labeled_mask = src_data_cluster[i_slice]["src_labeled_mask"]
        cur_roi_idx_to_label_offset = src_data_cluster[i_slice]["roi_idx_to_label_offset"]
        cur_slice_num = src_data_cluster[i_slice]["slice_num"]
        nof_rois = int(np.max(cur_labeled_mask))
        cur_roi_center_ijs = np.zeros((nof_rois,2))
        cur_roi_center_yxz_ums = np.zeros((nof_rois,3))
        for i_roi in range(nof_rois):
            roi_label = i_roi + cur_roi_idx_to_label_offset
            roi_mask = (cur_labeled_mask == roi_label)
            roi_mask_is, roi_mask_js = np.where(roi_mask)
            roi_mask_center_i = np.mean(roi_mask_is)
            roi_mask_center_j = np.mean(roi_mask_js)
            cur_roi_center_ijs[i_roi, 0] = roi_mask_center_i
            cur_roi_center_ijs[i_roi, 1] = roi_mask_center_j
            cur_roi_center_yxz_ums[i_roi,0] = image_ij_pixelsize_um[0] * roi_mask_center_i
            cur_roi_center_yxz_ums[i_roi,1] = image_ij_pixelsize_um[1] * roi_mask_center_j
            cur_roi_center_yxz_ums[i_roi,2] = image_z_stepsize_um * cur_slice_num
        src_data_cluster[i_slice]["roi_center_yxz_ums"] = cur_roi_center_yxz_ums


def add_mix_trial_avg_dFF(src_data_cluster):
    nof_slices = len(src_data_cluster)
    for i_slice in range(nof_slices):
        cur_stim_FmFnue_group_dFF = src_data_cluster[i_slice]["stim_FmFneu_group_dFF"]
        cur_blank_FmFnue_group_dFF = src_data_cluster[i_slice]["blank_FmFneu_group_dFF"]
        cur_mix_dFF = mix_dFF_group([cur_blank_FmFnue_group_dFF, cur_stim_FmFnue_group_dFF])
        cur_trial_avg_dFF = np.mean(cur_mix_dFF, axis = 2)
        cur_nof_rois = cur_trial_avg_dFF.shape[0]
        cur_trial_avg_dFF = cur_trial_avg_dFF.reshape((cur_nof_rois,-1))
        src_data_cluster[i_slice]["mix_trial_avg_dFF"] = cur_trial_avg_dFF


def add_continous_dFF(src_data_cluster, neuropil_subtract_rate = 0.7):
    nof_slices = len(src_data_cluster)

    F_len = src_data_cluster[0]["Fs"].shape[-1]
    Fneu_len = src_data_cluster[0]["Fneus"].shape[-1]
        
    nof_stims = src_data_cluster[0]["stim_tstamps"].shape[0]
    blank_tstamps = src_data_cluster[0]["blank_tstamps"]
    stim_tstamps = src_data_cluster[0]["stim_tstamps"]

    for i_slice in range(nof_slices):
        
        Fs = src_data_cluster[i_slice]["Fs"]
        Fneus = src_data_cluster[i_slice]["Fneus"]
        
        nof_rois = src_data_cluster[i_slice]["Fs"].shape[0]
        
        Fneus_group_F0 = src_data_cluster[i_slice]["Fneus_group_F0"]
        Fneu_continous_F0s = np.zeros((nof_rois, 1, Fneu_len))
        for i_stim in range(nof_stims):
            Fneu_continous_F0s[:, :, blank_tstamps[i_stim, 0]:stim_tstamps[i_stim,1]] = Fneus_group_F0[: ,i_stim, :, :]
        Fneu_continous_F0s[:, :, :blank_tstamps[0, 0]] = Fneus_group_F0[:, 0, :, :]
        Fneu_continous_F0s[:, :, stim_tstamps[-1,1]:] = Fneus_group_F0[:, -1, :, :]
        for i_stim in range(nof_stims - 1):
            fill_tstamps = np.arange(stim_tstamps[i_stim,1], blank_tstamps[i_stim+1, 0])
            fill_right_vals = (fill_tstamps - fill_tstamps[0])/(fill_tstamps[-1] - fill_tstamps[0]) * Fneus_group_F0[:,i_stim+1, :, :]
            fill_left_vals = (fill_tstamps[-1] - fill_tstamps)/(fill_tstamps[-1] - fill_tstamps[0]) * Fneus_group_F0[:,i_stim, :, :]
            Fneu_continous_F0s[:, :, fill_tstamps] = fill_left_vals + fill_right_vals
            # Fneu_continous_F0s[:, :, fill_tstamps] = Fneus_group_F0[:,i_stim, :, :]
        
        Fneus_continous_dFs = Fneus - Fneu_continous_F0s
        FmFneu_continous_Fs = Fs - neuropil_subtract_rate * Fneus_continous_dFs
        
        FmFneu_group_F0 = src_data_cluster[i_slice]["FmFneu_group_F0"]
        FmFneu_continous_F0s = np.zeros((nof_rois, 1, F_len))
        for i_stim in range(nof_stims):
            FmFneu_continous_F0s[:, :, blank_tstamps[i_stim, 0]:stim_tstamps[i_stim,1]] = FmFneu_group_F0[:, i_stim, :, :]
        FmFneu_continous_F0s[:, :, :blank_tstamps[0, 0]] = FmFneu_group_F0[:, 0, :, :]
        FmFneu_continous_F0s[:, :, stim_tstamps[-1,1]:] = FmFneu_group_F0[:, -1, :, :]
        for i_stim in range(nof_stims - 1):
            fill_tstamps = np.arange(stim_tstamps[i_stim,1], blank_tstamps[i_stim+1, 0])
            fill_right_vals = (fill_tstamps - fill_tstamps[0])/(fill_tstamps[-1] - fill_tstamps[0]) * FmFneu_group_F0[:, i_stim+1, :, :]
            fill_left_vals = (fill_tstamps[-1] - fill_tstamps)/(fill_tstamps[-1] - fill_tstamps[0]) * FmFneu_group_F0[:, i_stim, :, :]
            FmFneu_continous_F0s[:, :, fill_tstamps] = fill_left_vals + fill_right_vals
            # FmFneu_continous_F0s[:, :, fill_tstamps] = FmFneu_group_F0[:, i_stim, :, :]
        
        FmFneu_continous_dFFs = (Fs - FmFneu_continous_F0s)/FmFneu_continous_F0s
        FmFneu_continous_dFFs_trial_avg = np.mean(FmFneu_continous_dFFs, axis = 1)

        src_data_cluster[i_slice]["neuropil_subtract_rate"] = neuropil_subtract_rate
        src_data_cluster[i_slice]["Fneu_continous_F0s"] = Fneu_continous_F0s
        src_data_cluster[i_slice]["FmFneu_continous_F0s"] = FmFneu_continous_F0s
        src_data_cluster[i_slice]["FmFneu_continous_dFFs"] = FmFneu_continous_dFFs
        src_data_cluster[i_slice]["FmFneu_continous_dFFs_trial_avg"] = FmFneu_continous_dFFs_trial_avg


def add_roi_feature(src_data_cluster, feature_key):
    
    assert feature_key in src_data_cluster[0].keys()

    nof_slices = len(src_data_cluster)

    for i_slice in range(nof_slices):
        cur_labeled_mask = src_data_cluster[i_slice]["src_labeled_mask"]
        nof_rois = int(np.max(cur_labeled_mask))
        cur_features = src_data_cluster[i_slice][feature_key]
        assert cur_features.shape[0] == nof_rois
        if len(cur_features.shape) > 2:
            cur_features = cur_features.reshape((nof_rois, -1))
        src_data_cluster[i_slice]["roi_features"] = cur_features