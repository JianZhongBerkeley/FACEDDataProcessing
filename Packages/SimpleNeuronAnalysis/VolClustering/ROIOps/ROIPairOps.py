from http.client import NON_AUTHORITATIVE_INFORMATION
import numpy as np
from ..Math import DataPairMath 


NONE_MAPPING_IDX = -1


def connect_related_roi_pair(
        roi_pair_labeled_masks,
        roi_pair_roi_idx_to_label_offsets,
        roi_pair_features,        
):
    MASK_DIM = 2 

    nof_slices = len(roi_pair_labeled_masks)

    assert nof_slices == 2
    assert len(roi_pair_roi_idx_to_label_offsets) == nof_slices
    assert len(roi_pair_features) == nof_slices
    assert roi_pair_labeled_masks[0].shape == roi_pair_labeled_masks[1].shape
    assert len(roi_pair_labeled_masks[0].shape) == MASK_DIM 

    connected_roi_idx_maps = [None for _ in range(nof_slices)]
    for i_slice in range(nof_slices):
        cur_labeled_mask = roi_pair_labeled_masks[i_slice]
        nof_rois = int(np.max(cur_labeled_mask))
        init_map = np.full((nof_rois,), NONE_MAPPING_IDX, dtype = int)
        connected_roi_idx_maps[i_slice] = init_map

    for i_slice in range(nof_slices):
        cur_slice_idx = i_slice
        search_slice_idx = nof_slices - i_slice - 1

        cur_labeled_mask = roi_pair_labeled_masks[cur_slice_idx]
        search_labeled_mask = roi_pair_labeled_masks[search_slice_idx]

        cur_roi_idx_to_label_offset = roi_pair_roi_idx_to_label_offsets[cur_slice_idx]
        search_roi_idx_to_label_offset = roi_pair_roi_idx_to_label_offsets[search_slice_idx]

        cur_roi_features = roi_pair_features[cur_slice_idx]
        search_roi_features = roi_pair_features[search_slice_idx]

        cur_nof_rois = int(np.max(cur_labeled_mask))
        for i_cur_roi in range(cur_nof_rois):
            cur_roi_label = i_cur_roi + cur_roi_idx_to_label_offset
            search_candidate_roi_labels = np.unique(search_labeled_mask[cur_labeled_mask == cur_roi_label], axis = None)
            search_candidate_roi_labels = search_candidate_roi_labels[search_candidate_roi_labels > 0]
            search_candidate_roi_idxs = search_candidate_roi_labels - search_roi_idx_to_label_offset
            search_candidate_roi_idxs = search_candidate_roi_idxs.astype(int)

            if search_candidate_roi_idxs.size > 0:
                max_search_pcc = -2
                max_search_idx = -1
                max_search_roi_idx = NONE_MAPPING_IDX
                cur_roi_feature = cur_roi_features[i_cur_roi]
                for i_search_idx in range(search_candidate_roi_idxs.size):
                    i_search_roi_idx = search_candidate_roi_idxs[i_search_idx]
                    search_cand_feature = search_roi_features[i_search_roi_idx]
                    cur_pcc = DataPairMath.pearson_correlation_coeff([cur_roi_feature, search_cand_feature], axis = -1)
                    if cur_pcc > max_search_pcc:
                        max_search_pcc = cur_pcc
                        max_search_idx = i_search_idx
                        max_search_roi_idx = i_search_roi_idx
                connected_roi_idx_maps[cur_slice_idx][i_cur_roi] = max_search_roi_idx

    for i_slice in range(nof_slices):
        cur_slice_idx = i_slice
        search_slice_idx = nof_slices - i_slice - 1
        
        cur_connected_roi_idx_map = connected_roi_idx_maps[cur_slice_idx]
        search_connected_roi_idx_map = connected_roi_idx_maps[search_slice_idx]
        
        cur_nof_rois = cur_connected_roi_idx_map.size

        for i_cur_roi in range(cur_nof_rois):
            cur_roi_search_idx = cur_connected_roi_idx_map[i_cur_roi]
            if cur_roi_search_idx == NONE_MAPPING_IDX:
                continue
            if i_cur_roi != search_connected_roi_idx_map[cur_roi_search_idx]:
                connected_roi_idx_maps[cur_slice_idx][i_cur_roi] = NONE_MAPPING_IDX

    return connected_roi_idx_maps






