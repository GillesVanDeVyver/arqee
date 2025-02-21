import numpy as np
import arqee
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage.transform import resize
import cv2

QUALITY_COLORS = np.array([(0.929, 0.106, 0.141),  # not visible, red
                           (0.957, 0.396, 0.137),  # poor, orange
                           (1, 0.984, 0.090),  # ok, yellow
                           (0.553, 0.776, 0.098),  # good, light green
                           (0.09, 0.407, 0.216)])  # excellent, dark green
QUALITY_CLASSES = ['not visible', 'poor', 'ok', 'good', 'excellent']


def create_visualization(image, segmentation, labels=None, colors=None, remove_oos=False,contour=False,
                         eps=1e-6):
    '''
    Create a visualization of the ultrasound image with the segmentation overlayed
    :param image: ndarray
        ndarray with shape (width,height) containing the pixel values of the grayscale ultrasound image
    :param segmentation:  ndarray
        ndarray with shape (labels,width,height) containing the segmentation (one-hot encoded) or
        ndarray with shape (width,height) containing the segmentation (not one-hot encoded)
    :param labels: list of int, optional
        the labels of the segments to visualize
        If not specified, default value is [0, 1, 2, 3, 4, 5, 6, 7]
    :param colors: ndarray, optional
        ndarray with shape (n,3) containing the colors to use for the segments
        If not specified, default value is np.array([(1,0,0),(0,0,1),(0,1,0),(1,1,0),(0,1,1),(1,0,1),(1,1,1),
                                      (0.55,0.27,0.07),(1,0.55,0)])
    :param remove_oos
        remove out of sector parts
        If True, parts of the segmentation outside the sector will be removed
    :param eps: float
        small value to avoid division by zero
    :param contour: bool
        If True, the contours of each region will be plotted instead of the filled regions.
        The same colors will be used for the contours as for the filled regions.
    :return: ndarray
        the visualization of the ultrasound image with the segmentation overlayed with values in range [0,255]
    '''
    image_rescaled = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    if labels is None:
        labels = [0, 1, 2, 3, 4, 5, 6, 7]
    if colors is None:
        colors = np.array([(1, 0, 0), (0, 0, 1), (0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1),
                           (0.55, 0.27, 0.07), (1, 0.55, 0)])

    if len(labels) > len(colors):
        raise ValueError('Not enough colors for plotting')

    if len(segmentation.shape) == 3:
        one_hot = True
        if max(labels) > len(segmentation):
            raise ValueError('Labels should be in range of the number of channels in the segmentation')
        resize_shape = (segmentation.shape[1], segmentation.shape[2])
    else:
        one_hot = False
        resize_shape = (segmentation.shape[0], segmentation.shape[1])
    # resize image to the same size as the segmentation
    image_resized = resize(image_rescaled, resize_shape, preserve_range=True)
    oos_mask = image_resized < eps

    result = np.zeros((image_resized.shape[0], image_resized.shape[1], 3))
    for i in range(3):
        result[:, :, i] = image_resized / 255
    for i, label in enumerate(labels):
        if one_hot:
            label_semgentation = segmentation[label, ...] > 0.5
        else:
            label_semgentation = segmentation == label

        if contour:
            label_semgentation = label_semgentation.astype(np.uint8)
            contours, _ = cv2.findContours(label_semgentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, colors[i], 1)

        else:
            if colors[i, 0] != 0:
                result[label_semgentation, 0] = np.clip(colors[i, 0] * (0.35 +
                                                                        result[label_semgentation, 0]), 0.0, 1.0)
            if colors[i, 1] != 0:
                result[label_semgentation, 1] = np.clip(colors[i, 1] * (0.35 +
                                                                        result[label_semgentation, 1]), 0.0, 1.0)
            if colors[i, 2] != 0:
                result[label_semgentation, 2] = np.clip(colors[i, 2] * (0.35 +
                                                                        result[label_semgentation, 2]), 0.0, 1.0)
    if remove_oos:
        result[oos_mask] = 0 # set out of sector parts to black
    return (result * 255).astype(np.uint8)


def create_visualization_quality(image, segmentation, quality_labels, seg_labels=None, remove_oos=True,
                                 cmap='gray'):
    '''
    Create a visualization of the ultrasound image with the segmentation overlayed
    The colors of the regions is determined by the quality labels
    :param image: ndarray
        ndarray with shape (width,height) containing the pixel values of the grayscale ultrasound image
    :param segmentation: ndarray
        ndarray with shape (labels,width,height) containing the segmentation (one-hot encoded) or
        ndarray with shape (width,height) containing the segmentation (not one-hot encoded)
    :param quality_labels: list of int or str
        the quality labels of the segments to visualize
    :param seg_labels: list of int, optional
        the labels of the segments to visualize
        If not specified, default value is [0, 1, 2, 3, 4, 5, 6, 7]
    :param remove_oos
        remove out of sector parts
        If True, parts of the segmentation outside the sector will be removed
    :param cmap: str
        the colormap to use for the grayscale image
    :return: ndarray
        the visualization of the ultrasound image with the segmentation overlayed with values in range [0,255]
    '''
    if seg_labels is None:
        seg_labels = [0, 1, 2, 3, 4, 5, 6, 7]
    if len(quality_labels) != len(seg_labels):
        raise ValueError('quality_labels and seg_labels should have the same length')

    colors = []
    for quality, label in zip(quality_labels, seg_labels):
        if type(quality) == float or type(quality) == np.float64:
            quality = int(quality)
        if type(quality) == int or type(quality) == np.int64:
            if quality < 1:
                quality = 1
            if quality > 5:
                quality = 5
            quality_index = quality - 1
        elif type(quality) == str or type(quality) == np.str_:
            quality_index = arqee.categorical_to_numerical(quality) - 1
        else:
            raise ValueError('quality_labels should be a list of strings or integers')

        quality_color = QUALITY_COLORS[quality_index]
        colors.append(quality_color)

    colors = np.array(colors)

    return create_visualization(image, segmentation, seg_labels, colors,remove_oos=remove_oos)


def plot_visual_results_img(image, segmentation, quality_labels, seg_labels=None, clim=None):
    '''
    Plot the ultrasound image with the segmentation overlayed
    The colors of the regions are determined by the quality labels
    :param image: ndarray
        ndarray with shape (width,height) containing the pixel values of the grayscale ultrasound image
    :param segmentation: ndarray
        ndarray with shape (labels,width,height) containing the segmentation (one-hot encoded) or
        ndarray with shape (width,height) containing the segmentation (not one-hot encoded)
        The segmentation should already be divided into segments
    :param quality_labels: list of int or str
        the quality labels of the segments to visualize
    :param seg_labels: list of int, optional
        the labels of the segments to visualize
        If not specified, default value is [0, 1, 2, 3, 4, 5, 6, 7]
    :param clim: list of int, optional
        the min and max values of the grayscale image, if not specified, default value is [0, 255]
    :return: tuple
        tuple containing (fig, ax, visuzalization_quality) where
        - fig: matplotlib.figure.Figure
            the figure containing the plot
        - ax: ndarray
            ndarray containing the axis of the plot
        - visualization_quality: ndarray
            the visualization of the ultrasound image with the segmentation overlayed with values in range [0,255]
    '''
    if clim is None:
        clim = [0, 255]
    visuzalization_quality = create_visualization_quality(image, segmentation, quality_labels, seg_labels)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Plot original image
    ax[0].imshow(image, cmap=cmap)
    ax[0].set_title('Input B-Mode image')
    ax[0].axis('off')  # Turn off axis for original image

    # Plot visualization
    ax[1].imshow(visuzalization_quality, clim=clim)
    ax[1].set_title('Predicted quality')
    ax[1].axis('off')  # Turn off axis for visualization

    # Create a custom color bar
    fig.subplots_adjust(right=0.85)  # Adjust the position of the color bar

    # Create a color bar with custom colors and labels
    cmap_quality = ListedColormap(QUALITY_COLORS)
    bounds = range(len(QUALITY_COLORS) + 1)
    norm = plt.Normalize(bounds[0], bounds[-1])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap_quality), ax=ax, ticks=np.arange(len(QUALITY_COLORS)) + 0.5)
    cb.ax.set_yticklabels(QUALITY_CLASSES)

    return fig, ax, visuzalization_quality


def plot_quality_prediction_result(image, segmentation, quality_labels, seg_labels=None):
    '''
    Plot the ultrasound image with the segmentation overlayed colored according to the quality labels
    :param image: ndarray
        ndarray with shape (width,height) containing the pixel values of the grayscale ultrasound image
    :param segmentation: ndarray
        ndarray with shape (labels,width,height) containing the segmentation (one-hot encoded) or
        ndarray with shape (width,height) containing the segmentation (not one-hot encoded)
        The segmentation should not yet be divided into segments
    :param quality_labels: list of int or str
        the quality labels of the segments to visualize
    :param seg_labels: list of int, optional
        the labels of the segments to visualize
        If not specified, default value is [0, 1, 2, 3, 4, 5, 6, 7]
    :return: tuple
        tuple containing (fig, ax, visuzalization_quality) where
        - fig: matplotlib.figure.Figure
            the figure containing the plot
        - ax: ndarray
            ndarray containing the axis of the plot
        - visualization_quality: ndarray
            the visualization of the ultrasound image with the segmentation overlayed with values in range [0,255]
    '''
    divided_seg = arqee.divide_segmentation(segmentation)
    return plot_visual_results_img(image, divided_seg, quality_labels, seg_labels)
