import suite2p
import numpy as np


def suite2p_F_calculation(src_image_stack, src_labeled_mask, ops, get_mask = False):

    frame_height = src_image_stack.shape[1]
    frame_width = src_image_stack.shape[2]

    ops["Ly"] = frame_height
    ops["Lx"] = frame_width
    Lx = ops["Lx"]
    Ly = ops["Ly"]

    masks = src_labeled_mask
    f_reg = src_image_stack

    stat = []
    for _, u in enumerate(np.unique(masks)[1:]):
        ypix,xpix = np.nonzero(masks==u)
        npix = len(ypix)
        stat.append({'ypix': ypix, 'xpix': xpix, 'npix': npix, 'lam': np.ones(npix, np.float32), 'med': [np.mean(ypix), np.mean(xpix)]})
    stat = np.array(stat)
    stat = suite2p.detection.roi_stats(stat, Ly, Lx)  

    cell_masks, neuropil_masks = suite2p.extraction.masks.create_masks(stat, Ly, Lx, ops)

    stat_after_extraction, F, Fneu, F_chan2, Fneu_chan2 = suite2p.extraction_wrapper(stat, f_reg, f_reg_chan2 = None,ops=ops)

    result = (F, Fneu)
    if get_mask:
        result += (cell_masks, neuropil_masks)

    return result


def suite2p_F_neuropil_subtraction(F, dFneu, ops):
    F_m_Fneu = F - ops['neucoeff'] * dFneu
    return F_m_Fneu