def verify_ascending_order(src_data_cluster):
    nof_slices = len(src_data_cluster)
    nof_slice_pair = nof_slices - 1
    assert nof_slice_pair >= 0
    order_correct = True
    for i_slice_pair in range(nof_slice_pair):
        cur_slice = i_slice_pair
        next_slice = i_slice_pair + 1
        cur_slice_num = src_data_cluster[cur_slice]["slice_num"]
        next_slice_num = src_data_cluster[next_slice]["slice_num"]
        if cur_slice_num >  next_slice_num:
            order_correct = False
    return order_correct