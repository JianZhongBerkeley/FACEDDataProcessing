import numpy as np
import sklearn.cluster


from .Utils import calculate_distance
from ..Math import DataPairMath
from ..ROIOps import ROIPairOps


def remove_impossible_links(src_data_cluster, max_roi_distance_um, min_feature_pcc):
    nof_slices = len(src_data_cluster)
    nof_slice_pair = nof_slices - 1
    assert(nof_slice_pair > 0)

    for i_slice_pair in range(nof_slice_pair):
        cur_slice_idx = i_slice_pair
        next_slice_idx = i_slice_pair + 1

        cur_labeled_mask = src_data_cluster[cur_slice_idx]["src_labeled_mask"]
        next_labeled_mask = src_data_cluster[next_slice_idx]["src_labeled_mask"]

        cur_roi_idx_to_label_offset = src_data_cluster[cur_slice_idx]["roi_idx_to_label_offset"]
        next_roi_idx_to_label_offset = src_data_cluster[next_slice_idx]["roi_idx_to_label_offset"]

        cur_forward_roi_pair_link = src_data_cluster[cur_slice_idx]["forward_roi_pair_link"]
        next_backword_roi_pair_link = src_data_cluster[next_slice_idx]["backward_roi_pair_link"]
        
        cur_roi_features = src_data_cluster[cur_slice_idx]["roi_features"]
        next_roi_features = src_data_cluster[next_slice_idx]["roi_features"]

        cur_roi_center_yxz_ums = src_data_cluster[cur_slice_idx]["roi_center_yxz_ums"]
        next_roi_center_yxz_ums = src_data_cluster[next_slice_idx]["roi_center_yxz_ums"]
        
        cur_nof_roi = cur_forward_roi_pair_link.shape[0]
        for i_cur_roi in range(cur_nof_roi):
            i_next_roi = cur_forward_roi_pair_link[i_cur_roi]
            
            if i_next_roi < 0:
                continue
                
            cur_roi_center_yxz_um = cur_roi_center_yxz_ums[i_cur_roi,:]
            next_roi_center_yxz_um = next_roi_center_yxz_ums[i_next_roi,:]
            
            cur_roi_feature = cur_roi_features[i_cur_roi, :]
            next_roi_feature = next_roi_features[i_next_roi, :]

            distance = calculate_distance([cur_roi_center_yxz_um, next_roi_center_yxz_um], axis = -1)
            pcc = DataPairMath.pearson_correlation_coeff([cur_roi_feature, next_roi_feature], axis = -1)

            if pcc < min_feature_pcc or distance > max_roi_distance_um:
                src_data_cluster[cur_slice_idx]["forward_roi_pair_link"][i_cur_roi] = ROIPairOps.NONE_MAPPING_IDX
                src_data_cluster[next_slice_idx]["backward_roi_pair_link"][i_next_roi] = ROIPairOps.NONE_MAPPING_IDX

        assert np.sum(src_data_cluster[cur_slice_idx]["forward_roi_pair_link"] >= 0) == np.sum(src_data_cluster[next_slice_idx]["backward_roi_pair_link"] >= 0)


def clustering_roi_chain(src_data_cluster, max_roi_distance_um, min_feature_pcc):

    nof_slices = len(src_data_cluster)
    feature_len = src_data_cluster[0]["roi_features"].shape[-1]
    nof_stims = src_data_cluster[0]["stim_FmFneu_group_dFF"].shape[1]

    visited_roi = [None for _ in range(nof_slices)]
    for i_slice in range(nof_slices):
        cur_labeled_mask = src_data_cluster[i_slice]["src_labeled_mask"]
        cur_nof_rois = int(np.max(cur_labeled_mask))
        visited_roi[i_slice] = np.full((cur_nof_rois,), False)

    for i_slice in range(nof_slices):
        print(f"processing slice {i_slice} ...")
        cur_labeled_mask = src_data_cluster[i_slice]["src_labeled_mask"]
        nof_rois = int(np.max(cur_labeled_mask))
        for i_roi in range(nof_rois):
            if visited_roi[i_slice][i_roi]:
                continue
            cur_forward_roi_idx = i_roi
            cur_forward_slice_idx = i_slice
            forward_linked_roi_idxs = []
            forward_linked_slice_idxs = []

            while cur_forward_roi_idx >= 0:
                forward_linked_roi_idxs.append(cur_forward_roi_idx)
                forward_linked_slice_idxs.append(cur_forward_slice_idx)
                cur_forward_roi_pair_link = src_data_cluster[cur_forward_slice_idx]["forward_roi_pair_link"]
                if cur_forward_roi_pair_link is None:
                    cur_forward_roi_idx = -1
                    continue
                cur_forward_roi_idx = cur_forward_roi_pair_link[cur_forward_roi_idx]
                cur_forward_slice_idx = cur_forward_slice_idx + 1

            nof_linked_rois = len(forward_linked_roi_idxs)
            if len(forward_linked_roi_idxs) > 1:
                
                roi_chain_roi_center_yxz_ums = np.zeros((nof_linked_rois, 3))
                roi_chain_roi_features = np.zeros((nof_linked_rois, feature_len))
                roi_chain_roi_responses = np.zeros((nof_linked_rois, nof_stims))
                
                for i_roi_chain_idx in range(nof_linked_rois):
                    cur_roi_idx = forward_linked_roi_idxs[i_roi_chain_idx]
                    cur_slice_idx = forward_linked_slice_idxs[i_roi_chain_idx]

                    cur_roi_stim_FmFneu_group_dFF = src_data_cluster[cur_slice_idx]["stim_FmFneu_group_dFF"][cur_roi_idx,...]
                    cur_roi_response = np.mean(cur_roi_stim_FmFneu_group_dFF, axis = (1,2))
                    cur_roi_response = cur_roi_response/np.sum(cur_roi_response)
                    
                    roi_chain_roi_responses[i_roi_chain_idx] = cur_roi_response
                    roi_chain_roi_center_yxz_ums[i_roi_chain_idx] = src_data_cluster[cur_slice_idx]["roi_center_yxz_ums"][cur_roi_idx, :]
                    roi_chain_roi_features[i_roi_chain_idx] = src_data_cluster[cur_slice_idx]["roi_features"][cur_roi_idx, :]

                roi_chain_roi_center_yxz_scale = np.mean(np.std(roi_chain_roi_center_yxz_ums, axis = 0))
                roi_chain_roi_response_scale = np.mean(np.std(roi_chain_roi_responses, axis = 0))
                
                roi_chain_clustering_features = np.concatenate((roi_chain_roi_center_yxz_ums/roi_chain_roi_center_yxz_scale, roi_chain_roi_responses/roi_chain_roi_response_scale), axis = -1)

                roi_chain_labels = None
                for nof_clusters in range(1, nof_linked_rois + 1):
                    cur_kmean = sklearn.cluster.KMeans(n_clusters=nof_clusters, random_state=0, n_init="auto").fit(roi_chain_clustering_features)
                    roi_chain_labels = cur_kmean.labels_
                    cur_cluster_max_distance = 0
                    cur_cluster_min_pcc = 2
                    for i_label in range(nof_clusters):
                        cur_label_idxs = np.arange(nof_linked_rois)[roi_chain_labels == i_label]
                        for ii in range(len(cur_label_idxs)):
                            ii_roi_chain_idx = cur_label_idxs[ii]
                            for jj in range(ii + 1, len(cur_label_idxs)):
                                jj_roi_chain_idx = cur_label_idxs[jj]
                                cur_distance = calculate_distance([roi_chain_roi_center_yxz_ums[ii_roi_chain_idx,:],
                                                                roi_chain_roi_center_yxz_ums[jj_roi_chain_idx,:]],
                                                                axis = -1)
                                cur_pcc = DataPairMath.pearson_correlation_coeff([roi_chain_roi_features[ii_roi_chain_idx,:], 
                                                                                roi_chain_roi_features[jj_roi_chain_idx,:]], 
                                                                                axis = -1)
                                if cur_distance > cur_cluster_max_distance:
                                    cur_cluster_max_distance = cur_distance
                                if cur_pcc < cur_cluster_min_pcc:
                                    cur_cluster_min_pcc = cur_pcc
                                if cur_cluster_max_distance > max_roi_distance_um:
                                    break
                                if cur_cluster_min_pcc < min_feature_pcc:
                                    break
                            if cur_cluster_max_distance > max_roi_distance_um:
                                break
                            if cur_cluster_min_pcc < min_feature_pcc:
                                break
                        if cur_cluster_max_distance > max_roi_distance_um:
                            break
                        if cur_cluster_min_pcc < min_feature_pcc:
                            break
                    if cur_cluster_max_distance <= max_roi_distance_um and cur_cluster_min_pcc >= min_feature_pcc:
                        break

                for i_roi_chain_pair_idx in range(nof_linked_rois-1):
                    cur_roi_chain_idx = i_roi_chain_pair_idx
                    next_roi_chain_idx = i_roi_chain_pair_idx + 1
                    if roi_chain_labels[cur_roi_chain_idx] !=  roi_chain_labels[next_roi_chain_idx]:
                        cur_roi_idx = forward_linked_roi_idxs[cur_roi_chain_idx]
                        cur_slice_idx = forward_linked_slice_idxs[cur_roi_chain_idx]
                        next_roi_idx = forward_linked_roi_idxs[next_roi_chain_idx]
                        next_slice_idx = forward_linked_slice_idxs[next_roi_chain_idx]
                        src_data_cluster[cur_slice_idx]["forward_roi_pair_link"][cur_roi_idx] = ROIPairOps.NONE_MAPPING_IDX
                        src_data_cluster[next_slice_idx]["backward_roi_pair_link"][next_roi_idx] = ROIPairOps.NONE_MAPPING_IDX
            
            for i_roi_chain_idx in range(nof_linked_rois):
                cur_roi_idx = forward_linked_roi_idxs[i_roi_chain_idx]
                cur_slice_idx = forward_linked_slice_idxs[i_roi_chain_idx]
                visited_roi[cur_slice_idx][cur_roi_idx] = True