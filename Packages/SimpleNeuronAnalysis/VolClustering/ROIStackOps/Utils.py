import numpy as np
import re


def calculate_distance(location_pair, axis = -1):
    assert len(location_pair) == 2
    assert location_pair[0].shape == location_pair[1].shape
    distance = np.sqrt((np.sum((location_pair[0] - location_pair[1])**2, axis = axis)))
    return distance


def mix_dFF_group(dFF_groups):
    
    nof_groups = len(dFF_groups)
    nof_rois, nof_sessions, nof_trials, _ = dFF_groups[0].shape
    
    dFF_session_lens = [0 for _ in range(nof_groups)]
    tot_session_len = 0
    for i_group in range(nof_groups):
        cur_session_len = dFF_groups[i_group].shape[-1]
        dFF_session_lens[i_group] = cur_session_len
        tot_session_len += cur_session_len
        
    mix_dFF_group = np.zeros((nof_rois, nof_sessions, nof_trials, tot_session_len))
    
    session_start_idx = 0
    for i_group in range(nof_groups):
        cur_session_len = dFF_session_lens[i_group]
        mix_dFF_group[:,:,:,session_start_idx:session_start_idx + cur_session_len] = dFF_groups[i_group]
        session_start_idx = session_start_idx + cur_session_len

    return mix_dFF_group


def find_slice_num_from_str(src_str, slice_num_regex =  r"slice\d+"):
    slice_num = int(re.findall(r"\d+", 
                           re.findall(slice_num_regex, 
                                      src_str)[0]
                          )[0])
    return slice_num