# intensity_normalization
# https://github.com/jcreinhold/intensity-normalization

from __future__ import print_function, division
import os
import nibabel as nib
import logging
from intensity_normalization.errors import NormalizationError
import numpy as np
from scipy.interpolate import interp1d
try:
    from sklearn.mixture import GaussianMixture
except ImportError:
    from sklearn.mixture import GMM as GaussianMixture
from intensity_normalization.utilities.mask import gmm_class_mask
from functools import reduce
from operator import add
import ants
from scipy.sparse import bsr_matrix
from scipy.sparse.linalg import svds
from intensity_normalization.normalize.whitestripe import whitestripe, whitestripe_norm
from intensity_normalization.utilities import csf
from intensity_normalization.utilities import io, hist

logger = logging.getLogger(__name__)
# kde 방법
def kde_normalize(img, mask=None, contrast='t1', norm_value=1):
    if mask is not None:
        voi = img.get_data()[mask.get_data() == 1].flatten()
    else:
        voi = img.get_data()[img.get_data() > img.get_data().mean()].flatten()
    if contrast.lower() in ['t1', 'flair', 'last']:
        wm_peak = hist.get_last_mode(voi)
    elif contrast.lower() in ['t2', 'largest']:
        wm_peak = hist.get_largest_mode(voi)
    elif contrast.lower() in ['md', 'first']:
        wm_peak = hist.get_first_mode(voi)
    else:
        raise NormalizationError('Contrast {} not valid, needs to be `t1`,`t2`,`flair`,`md`,`first`,`largest`,`last`'.format(contrast))
    normalized = nib.Nifti1Image((img.get_data() / wm_peak) * norm_value,
                                 img.affine, img.header)
    return normalized

# nynl 방법




def nyul_normalize(img_dir, mask_dir=None, output_dir=None, standard_hist=None, write_to_disk=True):
    """
    Use Nyul and Udupa method ([1,2]) to normalize the intensities of a set of MR images

    Args:
        img_dir (str): directory containing MR images
        mask_dir (str): directory containing masks for MR images
        output_dir (str): directory to save images if you do not want them saved in
            same directory as data_dir
        standard_hist (str): path to output or use standard histogram landmarks
        write_to_disk (bool): write the normalized data to disk or nah

    Returns:
        normalized (np.ndarray): last normalized image from img_dir

    References:
        [1] N. Laszlo G and J. K. Udupa, “On Standardizing the MR Image
            Intensity Scale,” Magn. Reson. Med., vol. 42, pp. 1072–1081,
            1999.
        [2] M. Shah, Y. Xiao, N. Subbanna, S. Francis, D. L. Arnold,
            D. L. Collins, and T. Arbel, “Evaluating intensity
            normalization on MRIs of human brain with multiple sclerosis,”
            Med. Image Anal., vol. 15, no. 2, pp. 267–282, 2011.
    """
    input_files = [img_dir]
    if output_dir is None:
        out_fns = [None] * len(input_files)
    else:
        out_fns = []
        for fn in input_files:
            _, base, ext = io.split_filename(fn)
            out_fns.append(os.path.join(output_dir, base + '_hm' + ext))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    mask_files = [None] * len(input_files) if mask_dir is None else io.glob_nii(mask_dir)

    if standard_hist is None:
        logger.info('Learning standard scale for the set of images')
        standard_scale, percs = train(input_files, mask_files)
    elif not os.path.isfile(standard_hist):
        logger.info('Learning standard scale for the set of images')
        standard_scale, percs = train(input_files, mask_files)
        np.save(standard_hist, np.vstack((standard_scale, percs)))
    else:
        logger.info('Loading standard scale ({}) for the set of images'.format(standard_hist))
        standard_scale, percs = np.load(standard_hist)

    for i, (img_fn, mask_fn, out_fn) in enumerate(zip(input_files, mask_files, out_fns)):
        _, base, _ = io.split_filename(img_fn)
        logger.info('Transforming image {} to standard scale ({:d}/{:d})'.format(base, i+1, len(input_files)))
        img = io.open_nii(img_fn)
        mask = io.open_nii(mask_fn) if mask_fn is not None else None
        normalized = do_hist_norm(img, percs, standard_scale, mask)
        if write_to_disk:
            io.save_nii(normalized, out_fn, is_nii=True)

    return normalized


def get_landmarks(img, percs):
    """
    get the landmarks for the Nyul and Udupa norm method for a specific image

    Args:
        img (np.ndarray): image on which to find landmarks
        percs (np.ndarray): corresponding landmark percentiles to extract

    Returns:
        landmarks (np.ndarray): intensity values corresponding to percs in img
    """
    landmarks = np.percentile(img, percs)
    return landmarks


def train(img_fns, mask_fns=None, i_min=1, i_max=99, i_s_min=1, i_s_max=100, l_percentile=10, u_percentile=90, step=10):
    """
    determine the standard scale for the set of images

    Args:
        img_fns (list): set of NifTI MR image paths which are to be normalized
        mask_fns (list): set of corresponding masks (if not provided, estimated)
        i_min (float): minimum percentile to consider in the images
        i_max (float): maximum percentile to consider in the images
        i_s_min (float): minimum percentile on the standard scale
        i_s_max (float): maximum percentile on the standard scale
        l_percentile (int): middle percentile lower bound (e.g., for deciles 10)
        u_percentile (int): middle percentile upper bound (e.g., for deciles 90)
        step (int): step for middle percentiles (e.g., for deciles 10)

    Returns:
        standard_scale (np.ndarray): average landmark intensity for images
        percs (np.ndarray): array of all percentiles used
    """
    mask_fns = [None] * len(img_fns) if mask_fns is None else mask_fns
    percs = np.concatenate(([i_min], np.arange(l_percentile, u_percentile+1, step), [i_max]))
    standard_scale = np.zeros(len(percs))
    for i, (img_fn, mask_fn) in enumerate(zip(img_fns, mask_fns)):
        img_data = io.open_nii(img_fn).get_data()
        mask = io.open_nii(mask_fn) if mask_fn is not None else None
        mask_data = img_data > img_data.mean() if mask is None else mask.get_data()
        masked = img_data[mask_data > 0]
        landmarks = get_landmarks(masked, percs)
        min_p = np.percentile(masked, i_min)
        max_p = np.percentile(masked, i_max)
        f = interp1d([min_p, max_p], [i_s_min, i_s_max])
        landmarks = np.array(f(landmarks))
        standard_scale += landmarks
    standard_scale = standard_scale / len(img_fns)
    return standard_scale, percs


def do_hist_norm(img, landmark_percs, standard_scale, mask=None):
    """
    do the Nyul and Udupa histogram normalization routine with a given set of learned landmarks

    Args:
        img (nibabel.nifti1.Nifti1Image): image on which to find landmarks
        landmark_percs (np.ndarray): corresponding landmark points of standard scale
        standard_scale (np.ndarray): landmarks on the standard scale
        mask (nibabel.nifti1.Nifti1Image): foreground mask for img

    Returns:
        normalized (nibabel.nifti1.Nifti1Image): normalized image
    """
    img_data = img.get_data()
    mask_data = img_data > img_data.mean() if mask is None else mask.get_data()
    masked = img_data[mask_data > 0]
    landmarks = get_landmarks(masked, landmark_percs)
    f = interp1d(landmarks, standard_scale, fill_value='extrapolate')
    normed = f(img_data)
    return nib.Nifti1Image(normed, img.affine, img.header)

# gmm normalization


def gmm_normalize(img, brain_mask=None, norm_value=1, contrast='t1', bg_mask=None, wm_peak=None):
    """
    normalize the white matter of an image using a GMM to find the tissue classes

    Args:
        img (nibabel.nifti1.Nifti1Image): target MR image
        brain_mask (nibabel.nifti1.Nifti1Image): brain mask for img
        norm_value (float): value at which to place the WM mean
        contrast (str): MR contrast type for img
        bg_mask (nibabel.nifti1.Nifti1Image): if provided, use to zero bkgd
        wm_peak (float): previously calculated WM peak

    Returns:
        normalized (nibabel.nifti1.Nifti1Image): gmm wm peak normalized image
    """

    if wm_peak is None:
        wm_peak = gmm_class_mask(img, brain_mask=brain_mask, contrast=contrast)

    img_data = img.get_data()
    logger.info('Normalizing Data...')
    norm_data = (img_data/wm_peak)*norm_value
    norm_data[norm_data < 0.1] = 0.0
    
    if bg_mask is not None:
        logger.info('Applying background mask...')
        masked_image = norm_data * bg_mask.get_data()
    else:
        masked_image = norm_data

    normalized = nib.Nifti1Image(masked_image, img.affine, img.header)
    return normalized

def zscore_normalize(img, mask=None):
    """
    normalize a target image by subtracting the mean of the whole brain
    and dividing by the standard deviation

    Args:
        img (nibabel.nifti1.Nifti1Image): target MR brain image
        mask (nibabel.nifti1.Nifti1Image): brain mask for img

    Returns:
        normalized (nibabel.nifti1.Nifti1Image): img with WM mean at norm_value
    """

    img_data = img.get_data()
    if mask is not None and not isinstance(mask, str):
        mask_data = mask.get_data()
    elif mask == 'nomask':
        mask_data = img_data == img_data
    else:
        mask_data = img_data > img_data.mean()
    logical_mask = mask_data == 1  # force the mask to be logical type
    mean = img_data[logical_mask].mean()
    std = img_data[logical_mask].std()
    normalized = nib.Nifti1Image((img_data - mean) / std, img.affine, img.header)
    return normalized
# ws



def ws_normalize(img_dir, contrast, mask_dir=None, output_dir=None, write_to_disk=True):
    """
    Use WhiteStripe normalization method ([1]) to normalize the intensities of
    a set of MR images by normalizing an area around the white matter peak of the histogram

    Args:
        img_dir (str): directory containing MR images to be normalized
        contrast (str): contrast of MR images to be normalized (T1, T2, or FLAIR)
        mask_dir (str): if images are not skull-stripped, then provide brain mask
        output_dir (str): directory to save images if you do not want them saved in
            same directory as img_dir
        write_to_disk (bool): write the normalized data to disk or nah

    Returns:
        normalized (np.ndarray): last normalized image data from img_dir
            I know this is an odd behavior, but yolo

    References:
        [1] R. T. Shinohara, E. M. Sweeney, J. Goldsmith, N. Shiee,
            F. J. Mateen, P. A. Calabresi, S. Jarso, D. L. Pham,
            D. S. Reich, and C. M. Crainiceanu, “Statistical normalization
            techniques for magnetic resonance imaging,” NeuroImage Clin.,
            vol. 6, pp. 9–19, 2014.
    """

    # grab the file names for the images of interest
    data = [img_dir]

    # define and get the brain masks for the images, if defined
    if mask_dir is None:
        masks = [None] * len(data)
    else:
        masks = io.glob_nii(mask_dir)
        if len(data) != len(masks):
            raise NormalizationError('Number of images and masks must be equal, Images: {}, Masks: {}'
                                     .format(len(data), len(masks)))

    # define the output directory and corresponding output file names
    if output_dir is None:
        output_files = [None] * len(data)
    else:
        output_files = []
        for fn in data:
            _, base, ext = io.split_filename(fn)
            output_files.append(os.path.join(output_dir, base + '_ws' + ext))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    # do whitestripe normalization and save the results
    for i, (img_fn, mask_fn, output_fn) in enumerate(zip(data, masks, output_files), 1):
        logger.info('Normalizing image: {} ({:d}/{:d})'.format(img_fn, i, len(data)))
        img = io.open_nii(img_fn)
        mask = io.open_nii(mask_fn) if mask_fn is not None else None
        indices = whitestripe(img, contrast, mask=mask)
        normalized = whitestripe_norm(img, indices)
        if write_to_disk:
            logger.info('Saving normalized image: {} ({:d}/{:d})'.format(output_fn, i, len(data)))
            io.save_nii(normalized, output_fn)

    # output the last normalized image (mostly for testing purposes)
    return normalized


def whitestripe(img, contrast, mask=None, width=0.05, width_l=None, width_u=None):
    """
    find the "(normal appearing) white (matter) stripe" of the input MR image
    and return the indices

    Args:
        img (nibabel.nifti1.Nifti1Image): target MR image
        contrast (str): contrast of img (e.g., T1)
        mask (nibabel.nifti1.Nifti1Image): brainmask for img (None is default, for skull-stripped img)
        width (float): width quantile for the "white (matter) stripe"
        width_l (float): lower bound for width (default None, derives from width)
        width_u (float): upper bound for width (default None, derives from width)

    Returns:
        ws_ind (np.ndarray): the white stripe indices (boolean mask)
    """
    if width_l is None and width_u is None:
        width_l = width
        width_u = width
    img_data = img.get_data()
    if mask is not None:
        mask_data = mask.get_data()
        masked = img_data * mask_data
        voi = img_data[mask_data == 1]
    else:
        masked = img_data
        voi = img_data[img_data > img_data.mean()]
    if contrast.lower() in ['t1', 'last']:
        mode = hist.get_last_mode(voi)
    elif contrast.lower() in ['t2', 'flair', 'largest']:
        mode = hist.get_largest_mode(voi)
    elif contrast.lower() in ['md', 'first']:
        mode = hist.get_first_mode(voi)
    else:
        raise NormalizationError('Contrast {} not valid, needs to be `t1`,`t2`,`flair`,`md`,`first`,`largest`,`last`'.format(contrast))
    img_mode_q = np.mean(voi < mode)
    ws = np.percentile(voi, (max(img_mode_q - width_l, 0) * 100, min(img_mode_q + width_u, 1) * 100))
    ws_ind = np.logical_and(masked >= ws[0], masked <= ws[1])
    if len(ws_ind) == 0:
        raise NormalizationError('WhiteStripe failed to find any valid indices!')
    return ws_ind


def whitestripe_norm(img, indices):
    """
    use the whitestripe indices to standardize the data (i.e., subtract the
    mean of the values in the indices and divide by the std of those values)

    Args:
        img (nibabel.nifti1.Nifti1Image): target MR image
        indices (np.ndarray): whitestripe indices (see whitestripe func)

    Returns:
        norm_img (nibabel.nifti1.Nifti1Image): normalized image in nifti format
    """
    img_data = img.get_data()
    mu = np.mean(img_data[indices])
    sig = np.std(img_data[indices])
    norm_img_data = (img_data - mu)/sig
    norm_img = nib.Nifti1Image(norm_img_data, img.affine, img.header)
    return norm_img


# Ravel 방법

def ravel_normalize(img_dir, mask_dir, contrast, output_dir=None, write_to_disk=False,
                    do_whitestripe=True, b=1, membership_thresh=0.99, segmentation_smoothness=0.25,
                    do_registration=False, use_fcm=True, sparse_svd=False, csf_masks=False):
    """
    Use RAVEL [1] to normalize the intensities of a set of MR images to eliminate
    unwanted technical variation in images (but, hopefully, preserve biological variation)

    this function has an option that is modified from [1] in where no registration is done,
    the control mask is defined dynamically by finding a tissue segmentation of the brain and
    thresholding the membership at a very high level (this seems to work well and is *much* faster)
    but there seems to be some more inconsistency in the results

    Args:
        img_dir (str): directory containing MR images to be normalized
        mask_dir (str): brain masks for imgs (or csf masks if csf_masks is True)
        contrast (str): contrast of MR images to be normalized (T1, T2, or FLAIR)
        output_dir (str): directory to save images if you do not want them saved in
            same directory as data_dir
        write_to_disk (bool): write the normalized data to disk or nah
        do_whitestripe (bool): whitestripe normalize the images before applying RAVEL correction
        b (int): number of unwanted factors to estimate
        membership_thresh (float): threshold of membership for control voxels
        segmentation_smoothness (float): segmentation smoothness parameter for atropos ANTsPy
            segmentation scheme (i.e., mrf parameter)
        do_registration (bool): deformably register images to find control mask
        use_fcm (bool): use FCM for segmentation instead of atropos (may be less accurate)
        sparse_svd (bool): use traditional SVD (LAPACK) to calculate right singular vectors
            else use ARPACK
        csf_masks (bool): provided masks are the control masks (not brain masks)
            assumes that images are deformably co-registered

    Returns:
        Z (np.ndarray): unwanted factors (used in ravel correction)
        normalized (np.ndarray): set of normalized images from data_dir

    References:
        [1] J. P. Fortin, E. M. Sweeney, J. Muschelli, C. M. Crainiceanu,
            and R. T. Shinohara, “Removing inter-subject technical variability
            in magnetic resonance imaging studies,” Neuroimage, vol. 132,
            pp. 198–212, 2016.
    """
    img_fns = [img_dir]
    mask_fns = mask_dir

    if output_dir is None or not write_to_disk:
        out_fns = None
    else:
        out_fns = []
        for fn in img_fns:
            _, base, ext = io.split_filename(fn)
            out_fns.append(os.path.join(output_dir, base + ext))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    # get parameters necessary and setup the V array
    V, Vc = image_matrix(img_fns, contrast, masks=mask_fns, do_whitestripe=do_whitestripe,
                         return_ctrl_matrix=True, membership_thresh=membership_thresh,
                         do_registration=do_registration, smoothness=segmentation_smoothness,
                         use_fcm=use_fcm, csf_masks=csf_masks)

    # estimate the unwanted factors Z
    _, _, vh = np.linalg.svd(Vc, full_matrices=False) if not sparse_svd else \
               svds(bsr_matrix(Vc), k=b, return_singular_vectors='vh')
    Z = vh.T[:, 0:b]

    # perform the ravel correction
    V_norm = ravel_correction(V, Z)

    # save the results to disk if desired
    if write_to_disk:
        for i, (img_fn, out_fn) in enumerate(zip(img_fns, out_fns)):
            img = io.open_nii(img_fn)
            norm = V_norm[:, i].reshape(img.get_data().shape)
            io.save_nii(img, out_fn, data=norm)

    return Z, V_norm


def ravel_correction(V, Z):
    """
    correct the images (in the image matrix V) by removing the trend
    found in Z

    Args:
        V (np.ndarray): image matrix (rows are voxels, columns are images)
        Z (np.ndarray): unwanted factors (see ravel_normalize.py and the orig paper)

    Returns:
        res (np.ndarray): normalized images
    """
    means = np.mean(V, axis=1)  # row means
    beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(Z.T, Z)), Z.T), V.T)
    fitted = np.matmul(Z, beta).T  # this line (alone) gives slightly diff answer than R ver, otherwise exactly same
    res = V - fitted
    res = res + means[:,np.newaxis]
    return res


def image_matrix(imgs, contrast, masks=None, do_whitestripe=True, return_ctrl_matrix=False,
                 membership_thresh=0.99, smoothness=0.25, max_ctrl_vox=10000, do_registration=False,
                 ctrl_prob=1, use_fcm=False, csf_masks=False):
    """
    creates an matrix of images where the rows correspond the the voxels of
    each image and the columns are the images

    Args:
        imgs (list): list of paths to MR images of interest
        contrast (str): contrast of the set of imgs (e.g., T1)
        masks (list or str): list of corresponding brain masks or just one (template) mask
        do_whitestripe (bool): do whitestripe on the images before storing in matrix or nah
        return_ctrl_matrix (bool): return control matrix for imgs (i.e., a subset of V's rows)
        membership_thresh (float): threshold of membership for control voxels (want this very high)
            this option is only used if the registration is turned off
        smoothness (float): smoothness parameter for segmentation for control voxels
            this option is only used if the registration is turned off
        max_ctrl_vox (int): maximum number of control voxels (if too high, everything
            crashes depending on available memory) only used if do_registration is false
        do_registration (bool): register the images together and take the intersection of the csf
            masks (as done in the original paper, note that this takes much longer)
        ctrl_prob (float): given all data, proportion of data labeled as csf to be
            used for intersection (i.e., when do_registration is true)
        use_fcm (bool): use FCM for segmentation instead of atropos (may be less accurate)
        csf_masks (bool): provided masks are the control masks (not brain masks)
            assumes that images are deformably co-registered

    Returns:
        V (np.ndarray): image matrix (rows are voxels, columns are images)
        Vc (np.ndarray): image matrix of control voxels (rows are voxels, columns are images)
            Vc only returned if return_ctrl_matrix is True
    """
    img_shape = io.open_nii(imgs[0]).get_data().shape
    V = np.zeros((int(np.prod(img_shape)), len(imgs)))

    if return_ctrl_matrix:
        ctrl_vox = []

    masks = [None] * len(imgs)

    do_registration = do_registration and not csf_masks

    for i, (img_fn, mask_fn) in enumerate(zip(imgs, masks)):
        _, base, _ = io.split_filename(img_fn)
        img = io.open_nii(img_fn)

        mask_array = np.where(img.get_fdata() < 1 , img.get_fdata() ,1)
        mask = nib.Nifti1Image(mask_array, img.affine, img.header)

        # do whitestripe on the image before applying RAVEL (if desired)
        if do_whitestripe:
            logger.info('Applying WhiteStripe to image {} ({:d}/{:d})'.format(base, i + 1, len(imgs)))
            inds = whitestripe(img, contrast, mask)
            img = whitestripe_norm(img, inds)
        img_data = img.get_data()
        if img_data.shape != img_shape:
            raise NormalizationError('Cannot normalize because image {} needs to have same dimension '
                                     'as all other images ({} != {})'.format(base, img_data.shape, img_shape))
        V[:,i] = img_data.flatten()
        if return_ctrl_matrix:
            if do_registration and i == 0:
                logger.info('Creating control mask for image {} ({:d}/{:d})'.format(base, i + 1, len(imgs)))
                verbose = True if logger.getEffectiveLevel() == logging.getLevelName('DEBUG') else False
                ctrl_masks = []
                reg_imgs = []
                reg_imgs.append(csf.nibabel_to_ants(img))
                ctrl_masks.append(csf.csf_mask(img, mask, contrast=contrast, csf_thresh=membership_thresh,
                                               mrf=smoothness, use_fcm=use_fcm))
            elif do_registration and i != 0:
                template = ants.image_read(imgs[0])
                tmask = ants.image_read(masks[0])
                img = csf.nibabel_to_ants(img)
                logger.info('Starting registration for image {} ({:d}/{:d})'.format(base, i + 1, len(imgs)))
                reg_result = ants.registration(template, img, type_of_transform='SyN', mask=tmask, verbose=verbose)
                img = reg_result['warpedmovout']
                mask = csf.nibabel_to_ants(mask)
                reg_imgs.append(img)
                logger.info('Creating control mask for image {} ({:d}/{:d})'.format(base, i + 1, len(imgs)))
                ctrl_masks.append(csf.csf_mask(img, mask, contrast=contrast, csf_thresh=membership_thresh,
                                               mrf=smoothness, use_fcm=use_fcm))
            else:  # assume pre-registered
                logger.info('Finding control voxels for image {} ({:d}/{:d})'.format(base, i + 1, len(imgs)))
                ctrl_mask = csf.csf_mask(img, mask, contrast=contrast, csf_thresh=membership_thresh,
                                         mrf=smoothness, use_fcm=use_fcm) if csf_masks else mask.get_data()
                if np.sum(ctrl_mask) == 0:
                    raise NormalizationError('No control voxels found for image ({}) at threshold ({})'
                                             .format(base, membership_thresh))
                elif np.sum(ctrl_mask) < 100:
                    logger.warning('Few control voxels found ({:d}) (potentially a problematic image ({}) or '
                                   'threshold ({}) too high)'.format(int(np.sum(ctrl_mask)), base, membership_thresh))
                ctrl_vox.append(img_data[ctrl_mask == 1].flatten())

    if return_ctrl_matrix and not do_registration:
        min_len = min(min(map(len, ctrl_vox)), max_ctrl_vox)
        logger.info('Using {:d} control voxels'.format(min_len))
        Vc = np.zeros((min_len, len(imgs)))
        for i in range(len(imgs)):
            ctrl_voxs = ctrl_vox[i][:min_len]
            logger.info('Image {:d} control voxel stats -  mean: {:.3f}, std: {:.3f}'
                         .format(i+1, np.mean(ctrl_voxs), np.std(ctrl_voxs)))
            Vc[:,i] = ctrl_voxs
    elif return_ctrl_matrix and do_registration:
        ctrl_sum = reduce(add, ctrl_masks)  # need to use reduce instead of sum b/c data structure
        intersection = np.zeros(ctrl_sum.shape)
        intersection[ctrl_sum >= np.floor(len(ctrl_masks) * ctrl_prob)] = 1
        num_ctrl_vox = int(np.sum(intersection))
        Vc = np.zeros((num_ctrl_vox, len(imgs)))
        for i, img in enumerate(reg_imgs):
            ctrl_voxs = img.numpy()[intersection == 1]
            logger.info('Image {:d} control voxel stats -  mean: {:.3f}, std: {:.3f}'
                         .format(i+1, np.mean(ctrl_voxs), np.std(ctrl_voxs)))
            Vc[:,i] = ctrl_voxs
        del ctrl_masks, reg_imgs
        import gc; gc.collect()  # force a garbage collection, since we just used the majority of the system memory

    return V if not return_ctrl_matrix else (V, Vc)


def image_matrix_to_images(V, imgs):
    """
    convert an image matrix to a list of the correctly formated nifti images

    Args:
        V (np.ndarray): image matrix (rows are voxels, columns are images)
        imgs (list): list of paths to corresponding MR images in V

    Returns:
        img_list (list): list of nifti images extracted from V
    """
    img_list = []
    for i, img_fn in enumerate(imgs):
        img = io.open_nii(img_fn)
        nimg = nib.Nifti1Image(V[:, i].reshape(img.get_data().shape), img.affine, img.header)
        img_list.append(nimg)
    return img_list


# 실행
input_dir = r"C:\Users\POP\Desktop\work\MICCAI_BraTS_2019_Data_Training\HGG"
out_dir = r'C:\Users\POP\Desktop\work\MICCAI_BraTS_2019_Data_Training\change'
input_contrast = 'flair'

for (path, dir, files) in os.walk(input_dir):
    for file in files: 
        if input_contrast +'.nii' in file:
            img_dir = os.path.join(path, file)
            img = nib.load(img_dir)

            nib.save( zscore_normalize(img) , os.path.join(out_dir, 'zscore_normalize', file))
            ws_normalize(img_dir, input_contrast, output_dir=os.path.join(out_dir,'ws_normalize'), write_to_disk=True)
            ravel_normalize(img_dir ,mask_dir = None ,contrast = input_contrast , output_dir= os.path.join(out_dir,'ravel_normalize'), write_to_disk=True)
            nib.save(kde_normalize(img,contrast = input_contrast) , os.path.join(out_dir, 'kde_normalize', file))
            nyul_normalize(img_dir , output_dir= os.path.join(out_dir, 'nyul_normalize'))
            nib.save(gmm_normalize(img, contrast = input_contrast) , os.path.join(out_dir,'gmm_normalize' ,file))


