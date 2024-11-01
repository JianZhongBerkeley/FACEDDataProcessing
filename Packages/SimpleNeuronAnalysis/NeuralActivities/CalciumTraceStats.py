import numpy as np
from scipy import stats


def extract_trace_group(src_Fs, tstamps):
    nof_secs = tstamps.shape[0]
    nof_traces = src_Fs.shape[0]
    trace_len = np.abs(tstamps[0,0] - tstamps[0,1])
    F_group = np.zeros((nof_secs, nof_traces, trace_len))
    for i_sec in range(nof_secs):
        F_group[i_sec, :, :] = src_Fs[:,tstamps[i_sec,0] : tstamps[i_sec,1]]
        
    return F_group


def extract_roi_trace_group(src_roi_Fs, tstamps):
    nof_rois = src_roi_Fs.shape[0]
    nof_secs = tstamps.shape[0]
    nof_traces = src_roi_Fs.shape[1]
    trace_len = np.abs(tstamps[0,0] - tstamps[0,1])
    roi_F_group = np.zeros((nof_rois, nof_secs, nof_traces, trace_len))
    for i_roi in range(nof_rois):
        roi_F_group[i_roi, :, :, :] = extract_trace_group(src_roi_Fs[i_roi, :, :], tstamps)
    return roi_F_group


def calculate_response_group(blank_dFFs, stim_dFFs):
    nof_secs = stim_dFFs.shape[0]
    nof_traces= stim_dFFs.shape[1]
    
    response_group = np.zeros((nof_secs, 2, nof_traces))
    
    response_group[:,0,:] = np.mean(blank_dFFs, axis = -1)
    response_group[:,1,:] = np.mean(stim_dFFs, axis = -1)
    
    return response_group


def stim_step_t_test(response_group, test_steps = [1,0], alternative = "greater"):
    
    nof_orints = response_group.shape[0]
    test_results = []
    pvalues = np.zeros((nof_orints,))
    for i_orient in range(nof_orints):
        a = response_group[i_orient, test_steps[0], :]
        b = response_group[i_orient, test_steps[1], :] 
        result = stats.ttest_ind(a, b, alternative = alternative)
        test_results.append(result)
        pvalues[i_orient] = result.pvalue
    return pvalues, test_results
    

def stim_step_anova_oneway(response_group, test_step = 1):
    nof_orints = response_group.shape[0]
    comp_group = []
    for i_orint in range(nof_orints):
        comp_group.append(response_group[i_orint, test_step, :])
    result = stats.f_oneway(*comp_group)
    return result


def calculate_responses(response_group, stim_step_idx = 1):
    responses = np.mean(response_group[:,stim_step_idx,:], axis = -1)
    responses[responses < 0] = 0 
    return responses

