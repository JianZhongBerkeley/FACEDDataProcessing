import numpy as np


def double_gaussian(theta, R_offset, R_pref, R_oppo, theta_pref, sigma):
    return double_gaussian_periodic(theta, R_offset, R_pref, R_oppo, theta_pref, sigma)


def double_gaussian_periodic(theta, R_offset, R_pref, R_oppo, theta_pref, sigma):
    R_theta_pref = R_pref * np.exp(-((theta - theta_pref)**2)/(2*(sigma**2)))
    R_theta_pref_shift = R_pref * np.exp(-((theta - theta_pref - 2*np.pi )**2)/(2*(sigma**2)))
    R_theta_oppo = R_oppo * np.exp(-((theta - theta_pref - np.pi)**2)/(2*(sigma**2)))
    R_theta_oppo_shift = R_oppo * np.exp(-((theta - theta_pref + np.pi)**2)/(2*(sigma**2)))
    R_theta = R_offset + R_theta_pref + R_theta_oppo + R_theta_pref_shift + R_theta_oppo_shift
    return R_theta
