import numpy as np
import skimage
from scipy.stats import norm

def numerical_to_categorical(numerical_label):
    '''
    Convert numerical label to categorical string labels
    :param numerical_label: int
        Integer label
    :return: str
        String label
    '''
    if numerical_label <= 1.5:
        return 'not visible'
    elif numerical_label <= 2.5:
        return 'Poor'
    elif numerical_label <= 3.5:
        return 'Ok'
    elif numerical_label <= 4.5:
        return 'Good'
    else:
        return 'Excellent'


def categorical_to_numerical(category_label):
    '''
    Convert categorical label to numerical label
    :param category_label: str
        String label
    :return: int
        Integer label
    '''
    if category_label.lower() == 'not visible':
        return 1
    elif category_label.lower() == 'poor':
        return 2
    elif category_label.lower() == 'ok':
        return 3
    elif category_label.lower() == 'good':
        return 4
    elif category_label.lower() == 'excellent':
        return 5
    else:
        raise ValueError(f'Label {category_label} not recognized. Please provide a valid label')


def contrast_to_noise_ratio(foreground_pixels, background_pixels):
    '''
    Contrast to noise ratio or contrast to specle ratio, defined as
    (mu_1-mu_2)/sqrt(sigma_1^2+sigma_2^2), where
    - mu_1 is the mean pixel value of the foreground pixels
    - mu_2 is the mean pixel value of the background pixels
    - sigma_1 is the standard deviation of the pixel values of the foreground pixels
    - sigma_2 is the standard deviation of the pixel values of the background pixels
    First introduced in:
    Patterson, M. S., and F. S. Foster.
    "The improvement and quantitative assessment of B-mode images produced by an annular array/cone hybrid."
     Ultrasonic Imaging 5.3 (1983): 195-213.
    :param foreground_pixels: np.array
        The pixel values of the foreground pixels
    :param background_pixels: np.array
        The pixel values of the background pixels
    :return: float
        contrast to noise ratio
    '''
    return (np.mean(foreground_pixels) - np.mean(background_pixels)) / np.sqrt(
        np.var(foreground_pixels) + np.var(background_pixels))


def get_gcnr_from_prob_density(discete_prob_background,
                               discete_prob_segment,
                               verbose=False):
    '''
    Calculate the Generalised Contrast-to-Noise Ratio (GCNR) using the given discrete probability density functions
    of the segment and the background.
    :param discete_prob_background: ndarray
        discrete probability density function for the background
    :param discete_prob_segment: ndarray
        discrete probability density function for the segment
    :param verbose: bool, optional
        if True, print the overlap area (OVL)
    :return: float
        generalised CNR for the given discrete probability density functions
    '''
    if len(discete_prob_segment)==0:
        return np.nan
    pmin = 0.0
    for value_background,value_segment in zip(discete_prob_background,discete_prob_segment):
        pmin += min(value_background,value_segment)
    ovl = pmin/2
    if verbose:
        print('OVL: ' +str(ovl))
    return 1-ovl

def get_discrete_prob_density_funcs(pixels_foreground,
                                    pixels_background,
                                    max_value=255):
    '''
    Calculate the discrete probability density functions for the segment and the background
    :param pixels_foreground: ndarray
        pixel values of the region of interest
    :param pixels_background: ndarray
        pixel values of the background region
    :param max_value: int, optional
        maximum pixel value
    :return: tuple
        discrete probability density function for the background,
        number of values in the background,
        discrete probability density function for the segment,
        number of values in the segment
    '''
    discete_prob_background = np.array([0 for _ in range(0,max_value+1)])
    discete_prob_segment = np.array([0 for _ in range(0,max_value+1)])
    nb_values_background = 0
    nb_values_segment = 0
    for value in pixels_background:
        discete_prob_background[value] += 1
        nb_values_background += 1
    for value in pixels_foreground:
        discete_prob_segment[value] += 1
        nb_values_segment += 1
    discete_prob_background = discete_prob_background/nb_values_background
    discete_prob_segment = discete_prob_segment/nb_values_segment
    return discete_prob_background,nb_values_background,discete_prob_segment

def gcnr(pixels_foreground,
         pixels_background,
         max_value=255,
         verbose=False):
    '''
    Calculate the Generalised Contrast-to-Noise Ratio (GCNR) using the given pixel values of the region of interest
    and the background region.
    Based on
    A. Rodriguez-Molares, O. M. Hoel Rindal, J. D'hooge, S. -E. Måsøy, A. Austeng and H. Torp,
    "The Generalized Contrast-to-Noise Ratio," 2018 IEEE International Ultrasonics Symposium (IUS), 2018, pp. 1-4, doi:
    10.1109/ULTSYM.2018.8580101. https://ieeexplore.ieee.org/document/8580101
    :param pixels_foreground: ndarray
        pixel values of the region of interest
    :param pixels_background: ndarray
        pixel values of the background region
    :param max_value: int, optional
        maximum pixel value
    :param verbose: bool, optional
        if True, print the overlap area (OVL) of the discrete probability density functions
    :return: float
        generalised CNR for the given pixel values
    '''
    discrete_prob_background,nb_values_background,discrete_prob_segment=\
        get_discrete_prob_density_funcs(pixels_foreground,pixels_background,max_value=max_value)
    gcnr = get_gcnr_from_prob_density(discrete_prob_background,
                                      discrete_prob_segment,
                                      verbose=verbose)
    return gcnr


def normalize_image(img,max_pixel_value=255,mean=127.0,std_dev=32.0):
    '''
    This function normalizes the pixel values of the given image by applying histogram matching to
    a Gaussian distribution with the given mean and standard deviation.
    :param img: ndarray
        grayscale b-mode image as a ndarray with shape (width,height) and integer pixel values
        in the range [0, max_pixel_value].
    :param max_pixel_value: int, optional
        maximum pixel value of the image
    :param std_dev: int, optional
        standard deviation of the Gaussian distribution
    :param mean: int, optional
        mean of the Gaussian distribution
    :return: ndarray
        normalized image with shape (width,height).
    '''
    img = img.astype(np.uint8)
    # Mean and standard deviation of the Gaussian distribution
    values = np.arange(0, max_pixel_value + 1)
    # Compute the cumulative distribution function (CDF) for each value in the pixel range
    cdf_values_norm = norm.cdf(values, mean, std_dev)
    # get cdf of input image
    # filter out values outside the pixel range
    img = np.clip(img, 0, max_pixel_value)
    # get the cumulative distribution of the pixel values
    cdf_values_img, bin_centers = skimage.exposure.cumulative_distribution(img)
    # This distribution will only include pixel values present in the image
    # Extend the cumulative distribution to cover the entire range [0, max_pixel_value]
    start_bin = bin_centers[0]
    while start_bin > 0:
        cdf_values_img = np.insert(cdf_values_img, 0, 0)
        start_bin -= 1
    end_bin = bin_centers[-1]
    while end_bin < max_pixel_value:
        cdf_values_img = np.append(cdf_values_img, 1)
        end_bin += 1
    mapping = np.zeros(max_pixel_value+1, dtype=np.uint8)
    for i in range(max_pixel_value+1):
        # get the nearest value in the cumulative distribution of the template image
        diff = np.abs(cdf_values_img[i] - cdf_values_norm)
        mapping[i] = np.argmin(diff)
    # Apply mapping to source image
    matched_img = mapping[img]
    return matched_img


def wma_segmentation(seg_sequence, window_size):
    '''
    Perform weighted moving average on the given segmentation sequence with the given window size.
    :param seg_sequence: ndarray
        segmentation sequence as a ndarray with shape (nb_frames, nb_classes, height, width)
    :param window_size: int
        window size for the moving average
    :return: ndarray
        smoothed segmentation sequence with shape (nb_frames, nb_classes, height, width)
    '''
    nb_frames,nb_classes, height, width = seg_sequence.shape
    smoothed_sequence = np.zeros((nb_frames, nb_classes,height, width))

    for i in range(nb_frames):
        start = max(0, i - window_size)
        end = min(nb_frames, i + window_size + 1)

        # Symmetric distance-based weighting
        distances = np.abs(np.arange(start, end) - i)  # Distance from center
        weights = 1 / (distances + 1)  # Closer frames get higher weights
        weights = weights / np.sum(weights)  # Normalize

        smoothed_sequence[i] = np.average(seg_sequence[start:end], axis=0, weights=weights)

    return smoothed_sequence






















