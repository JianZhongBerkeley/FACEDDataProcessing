# FILE: CreateROIChain.py
# DESCRIPTION: functions to create roi chain connecting related ROIs
# @author: Jian Zhong

import numpy as np
from ..ROIOps import ROIPairOps


def create_roi_chain(src_data_cluster):
    
    nof_slices = len(src_data_cluster)
    nof_slice_pair = nof_slices - 1
    assert(nof_slice_pair > 0)

    for i_slice in range(nof_slices):
        src_data_cluster[i_slice]["forward_roi_pair_link"] = None
        src_data_cluster[i_slice]["backward_roi_pair_link"] = None

    for i_slice_pair in range(nof_slice_pair):
        cur_slice_idx = i_slice_pair
        next_slice_idx = i_slice_pair + 1

        cur_labeled_mask = src_data_cluster[cur_slice_idx]["src_labeled_mask"]
        next_labeled_mask = src_data_cluster[next_slice_idx]["src_labeled_mask"]

        cur_roi_idx_to_label_offset = src_data_cluster[cur_slice_idx]["roi_idx_to_label_offset"]
        next_roi_idx_to_label_offset = src_data_cluster[next_slice_idx]["roi_idx_to_label_offset"]

        cur_roi_features = src_data_cluster[cur_slice_idx]["roi_features"]
        next_roi_features = src_data_cluster[next_slice_idx]["roi_features"]

        connected_roi_pair_idx_map = ROIPairOps.connect_related_roi_pair(
            [cur_labeled_mask, next_labeled_mask],
            [cur_roi_idx_to_label_offset, next_roi_idx_to_label_offset],
            [cur_roi_features, next_roi_features],
        )

        assert np.sum(connected_roi_pair_idx_map[0] >= 0) == np.sum(connected_roi_pair_idx_map[1] >= 0)

        src_data_cluster[cur_slice_idx]["forward_roi_pair_link"] = connected_roi_pair_idx_map[0]
        src_data_cluster[next_slice_idx]["backward_roi_pair_link"] = connected_roi_pair_idx_map[1]


