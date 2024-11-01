import h5py
import os
import numpy as np

from .Utils import find_slice_num_from_str


def create_data_cluster_from_hdf5(src_hdf5_file_paths, src_slice_num_regex):
    src_data_file_paths = src_hdf5_file_paths
    
    nof_data_files = len(src_data_file_paths)

    src_data_cluster = [dict() for _ in range(nof_data_files)]

    for i_file in range(nof_data_files):
        src_data_file_path = src_data_file_paths[i_file]
        
        src_data_cluster[i_file]["list_idx"] = i_file
        src_data_cluster[i_file]["src_data_file_path"] = src_data_file_path
        
        src_data_file_name = os.path.split(src_data_file_path)[-1]
        src_data_cluster[i_file]["slice_num"] = find_slice_num_from_str(src_data_file_name, src_slice_num_regex)
        
        with h5py.File(src_data_file_path, "r") as hdf5_file:
            roi_idx_to_label_offset = hdf5_file["roi_idx_to_label_offset"][()]
            src_labeled_mask = hdf5_file["src_labeled_mask"][()]
            nof_rois = int(np.max(src_labeled_mask))

            init_acc_time_ms = hdf5_file["init_acc_time_ms"][()]
            time_per_slice_ms = hdf5_file["time_per_slice_ms"][()]
            static_moving_t_s = hdf5_file["static_moving_t_s"][()]
            stim_tstamp_shifted_s = hdf5_file["stim_tstamp_shifted_s"][()]
            stim_tstamp_shifted = hdf5_file["stim_tstamp_shifted"][()]
            ms_to_s = hdf5_file["ms_to_s"][()]
            s_to_ms = hdf5_file["s_to_ms"][()]

            nof_orient = hdf5_file["nof_orient"][()]
            orient_angles_deg = hdf5_file["orient_angles_deg"][()]
            orient_angles_rad = hdf5_file["orient_angles_rad"][()]
            
            Fs = hdf5_file["Fs"][()]
            Fneus = hdf5_file["Fneus"][()]
            stim_Fs_group = hdf5_file["stim_Fs_group"][()]
            blank_Fs_group = hdf5_file["blank_Fs_group"][()]
            stim_Fneus_group = hdf5_file["stim_Fneus_group"][()]
            blank_Fneus_group = hdf5_file["blank_Fneus_group"][()]
            Fneus_group_F0 = hdf5_file["Fneus_group_F0"][()]
            stim_FmFneu_group = hdf5_file["stim_FmFneu_group"][()]
            blank_FmFneu_group = hdf5_file["blank_FmFneu_group"][()]
            FmFneu_group_F0 = hdf5_file["FmFneu_group_F0"][()]
            stim_FmFneu_group_dFF = hdf5_file["stim_FmFnue_group_dFF"][()]
            blank_FmFneu_group_dFF = hdf5_file["blank_FmFnue_group_dFF"][()]

            t_test_pass_mask = hdf5_file["t_test_pass_mask"][()]
            anova_test_pass_mask = hdf5_file["anova_test_pass_mask"][()]
            valid_neuron_mask = hdf5_file["valid_neuron_mask"][()]
            gOSI_group = hdf5_file["gOSI_group"][()]

            blank_tstamps = hdf5_file["blank_tstamps"][()]
            stim_tstamps = hdf5_file["stim_tstamps"][()]
            
            src_data_cluster[i_file]["roi_idx_to_label_offset"] = roi_idx_to_label_offset
            src_data_cluster[i_file]["src_labeled_mask"] = src_labeled_mask
            src_data_cluster[i_file]["nof_rois"] = nof_rois
            
            src_data_cluster[i_file]["init_acc_time_ms"] = init_acc_time_ms
            src_data_cluster[i_file]["time_per_slice_ms"] = time_per_slice_ms
            src_data_cluster[i_file]["static_moving_t_s"] = static_moving_t_s
            src_data_cluster[i_file]["stim_tstamp_shifted_s"] = stim_tstamp_shifted_s
            src_data_cluster[i_file]["stim_tstamp_shifted"] = stim_tstamp_shifted
            src_data_cluster[i_file]["ms_to_s"] = ms_to_s
            src_data_cluster[i_file]["s_to_ms"] = s_to_ms

            src_data_cluster[i_file]["nof_orient"] = nof_orient
            src_data_cluster[i_file]["orient_angles_deg"] = orient_angles_deg
            src_data_cluster[i_file]["orient_angles_rad"] = orient_angles_rad
            
            src_data_cluster[i_file]["Fs"] = Fs
            src_data_cluster[i_file]["Fneus"] = Fneus
            src_data_cluster[i_file]["stim_Fs_group"] = stim_Fs_group
            src_data_cluster[i_file]["blank_Fs_group"] = blank_Fs_group
            src_data_cluster[i_file]["stim_Fneus_group"] = stim_Fneus_group
            src_data_cluster[i_file]["blank_Fneus_group"] = blank_Fneus_group
            src_data_cluster[i_file]["Fneus_group_F0"] = Fneus_group_F0
            src_data_cluster[i_file]["stim_FmFneu_group"] = stim_FmFneu_group
            src_data_cluster[i_file]["blank_FmFneu_group"] = blank_FmFneu_group
            src_data_cluster[i_file]["FmFneu_group_F0"] = FmFneu_group_F0
            src_data_cluster[i_file]["stim_FmFneu_group_dFF"] = stim_FmFneu_group_dFF
            src_data_cluster[i_file]["blank_FmFneu_group_dFF"] = blank_FmFneu_group_dFF

            src_data_cluster[i_file]["t_test_pass_mask"] = t_test_pass_mask
            src_data_cluster[i_file]["anova_test_pass_mask"] = anova_test_pass_mask
            src_data_cluster[i_file]["valid_neuron_mask"] = valid_neuron_mask
            src_data_cluster[i_file]["gOSI_group"] = gOSI_group

            src_data_cluster[i_file]["blank_tstamps"] = blank_tstamps
            src_data_cluster[i_file]["stim_tstamps"] = stim_tstamps

    return src_data_cluster