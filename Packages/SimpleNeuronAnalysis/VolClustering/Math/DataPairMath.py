import numpy as np


def convariance(data_pair, axis = -1):
    assert len(data_pair) == 2
    assert data_pair[0].shape == data_pair[1].shape
    
    nof_data = len(data_pair)
    
    src_shape = data_pair[0].shape
    dst_shape = np.array(src_shape)
    dst_shape[axis] = 1
    dst_shape = tuple(dst_shape)

    data_means = [None for _ in range(nof_data)]

    for i_data in range(nof_data):
        cur_data = data_pair[i_data]
        cur_mean = np.mean(cur_data, axis = axis)
        data_means[i_data] = cur_mean.reshape(dst_shape)

    covarience = np.mean( (data_pair[0] - data_means[0])*(data_pair[1] - data_means[1]), axis = axis)

    return covarience

def pearson_correlation_coeff(data_pair, axis = -1):
    assert len(data_pair) == 2
    assert data_pair[0].shape == data_pair[1].shape

    nof_data = len(data_pair)
    
    src_shape = data_pair[0].shape
    dst_shape = np.array(src_shape)
    dst_shape[axis] = 1
    dst_shape = tuple(dst_shape)

    data_stds = np.zeros(nof_data)
    for i_data in range(nof_data):
        cur_data = data_pair[i_data]
        cur_std = np.std(cur_data, axis = axis)
        data_stds[i_data] = cur_std.reshape(dst_shape)

    conv = convariance(data_pair, axis)
    
    pcc = conv/(data_stds[0] * data_stds[1])

    return pcc