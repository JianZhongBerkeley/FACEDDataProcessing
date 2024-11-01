import numpy as np
import scipy.ndimage
import skimage.filters


def fov_avg_illum_profile(src_image, illum_est_percentile, nof_fov, signal_est_gauss_sigma = 50, mode = "none"):
    src_image_height, src_image_width = src_image.shape
    fov_stack_width = src_image_width//nof_fov
    fov_stack = src_image.reshape((src_image_height, nof_fov, fov_stack_width))
    fov_stack_proj = np.mean(fov_stack, axis = 1)
    fov_illum_profile = np.percentile(fov_stack_proj, illum_est_percentile, axis = 0)
    illum_profile = np.tile(fov_illum_profile, (nof_fov,))

    src_linesignal_est = np.mean(src_image, axis = 0)
    src_linesignal_est = scipy.ndimage.gaussian_filter(src_linesignal_est, signal_est_gauss_sigma)
    
    illum_profile[illum_profile < 1] = 1
    if mode == "inv":
        illum_profile = illum_profile/src_linesignal_est
    else:
        illum_profile = illum_profile * src_linesignal_est
        illum_profile = illum_profile/np.mean(illum_profile)

    illum_profile[illum_profile < 0.1] = 0.1
    
    return illum_profile


def percentile_illum_profile(src_image, bkg_est_percentile = 1, line_est_percentile = 50, signal_est_gauss_sigma = 200, mode = "none"):
    src_image = src_image.astype(float)
    bkg_pxl_val = np.percentile(src_image, bkg_est_percentile)

    src_lineillum_est = np.percentile(src_image, line_est_percentile, axis = 0)
    src_lineillum_est[src_lineillum_est < bkg_pxl_val] = bkg_pxl_val
    src_lineillum_est.reshape((1,-1))

    src_linesignal_est = np.mean(src_image, axis = 0)
    src_linesignal_est = scipy.ndimage.gaussian_filter(src_linesignal_est, signal_est_gauss_sigma)

    src_lineillum_est[src_lineillum_est < 1] = 1

    illum_profile = src_lineillum_est
    if mode == "inv":
        illum_profile = illum_profile/src_linesignal_est
    else:
        illum_profile = illum_profile * src_linesignal_est
        illum_profile = illum_profile/np.mean(illum_profile)

    illum_profile[illum_profile < 0.1] = 0.1

    return illum_profile


def illum_norm_image(src_image, illum_est_func, params):
    illum_profile = illum_est_func(src_image , *params)
    dst_image = src_image/illum_profile.reshape((1,-1))
    dst_image = (np.mean(src_image)/np.mean(dst_image))*dst_image
    return dst_image


def illum_est_loss(src_image, illum_est_func, params):
    dst_image = illum_norm_image(src_image, illum_est_func, params)

    dst_image_edge = skimage.filters.sobel_v(dst_image)
    loss = np.mean(np.abs(np.mean(dst_image_edge, axis = 0)))
    
    return loss