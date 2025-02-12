import onnxruntime
import numpy as np
import os
import warnings
import arqee.CONST as CONST
from skimage.transform import resize
from tqdm import tqdm
import arqee
import abc


def apply_linear_model(inference_output, slope_intercept):
    '''
    Apply the linear model to the inference output. The linear model is defined by the slope and intercept.
    :param inference_output: np.array
        Numerical output of regression model as np.array
    :param slope_intercept: nd.array
        Array of length 2 containing (slope, intercept) as floats
        The slope is the slope of the linear model fitted to the validation output and labels
        The intercept is the intercept of the linear model fitted to the validation output and labels
    :return np.array
        Corrected inference output
    '''
    slope = slope_intercept[0]
    intercept = slope_intercept[1]
    return (inference_output - intercept) / slope


def pre_process_batch_generic(batch_data, verbose=True):
    '''
    Generic pre-processing to the correct input format before inference for batches
    :param batch_data: np.array
        Input data as np.array of shape (batch_size, channels, height, width
    :param verbose: bool
        If True, warnings will be printed concerning resizing and channels
    :return: np.array
        Pre-processed input data as np.array of shape (batch_size, channels, height, width
    '''
    # check if data has correct shape
    inference_input = batch_data.copy()
    if len(inference_input.shape) != 4:
        raise ValueError(
            'Expected data to have shape (batch_size, channels, 256, 256), got: ' + str(inference_input.shape) +
            '.\n Please make sure to provide the data in the correct format before running inference_batch.')
    if inference_input.shape[1] != 1:
        if verbose:
            warnings.warn(
                'Warning...........: Expected data to have 1 grayscale channel, got: ' + str(inference_input.shape[1]) +
                'channels.\n Only the first channel will be used for inference.')
        inference_input = inference_input[:, 0, :, :]
    if inference_input.shape[2] != 256 or inference_input.shape[3] != 256:
        if verbose:
            print('The given data has shape: ' + str(inference_input.shape) +
                  '.\n The data will be resized to (batch_size, 1, 256, 256) before running inference.')
        inference_input = resize(inference_input, (inference_input.shape[0], inference_input.shape[1], 256, 256),
                                 preserve_range=True)
    # convert data to float32
    inference_input = inference_input.astype(np.float32)
    return inference_input


def inference_batch_generic(model_object, inference_input, verbose=True):
    '''
    Perform inference on the given batch of data using the given onnx session
    :param inference_input: np.array
        Pre-processed input data as np.array of shape (batch_size, channels, height, width
    :param model_object: ONNXModelObject
        The model object containing the onnx session
        Only onnx is supported
    :param verbose: bool
        If True, info and warnings will be printed
    :return: np.array
        Inference output as np.array with either numerical or string labels
        The output of the inference is a ndarray with size bsx8 with the quality labels in the following order:
        basal_left,mid_left,apical_left,apical_right,mid_right,basal_right,annulus_left,annulus_right
    '''
    onnx_sess = model_object.session
    input_name = onnx_sess.get_inputs()[0].name
    output_name = onnx_sess.get_outputs()[0].name
    res = onnx_sess.run([output_name], {input_name: inference_input})[0]
    return res


def pre_process_img_generic(img_data, verbose=True):
    '''
    Generic pre-processing to the correct input format before inference for single images
    :param img_data: np.array
        Input data as np.array of shape (channels, height, width)
    :param verbose: bool
        If True, warnings will be printed concerning resizing and channels
    :return: np.array
        Pre-processed input data as np.array of shape (1, channels, height, width)
    '''
    if len(img_data.shape) != 3:
        raise ValueError('Expected img to have shape (channels, height, width), got: ' + str(img_data.shape) +
                         '.\n Please make sure to provide the img in the correct format before running inference_single_img.')
    inference_input = np.expand_dims(img_data, axis=0)
    return pre_process_batch_generic(inference_input, verbose)


def inference_img_generic(model_object, inference_input, verbose=True):
    '''
    Perform inference on the given data using the given onnx session
    :param model_object: ONNXModelObject
        The model object containing the onnx session
        Only onnx is supported
    :param inference_input: np.array
        Pre-processed input data as np.array of shape (channels, height, width)
    :param verbose: bool
        If True, info and warnings will be printed
    :return: np.array
        Inference output as np.array
    '''
    res_batch = inference_batch_generic(model_object, inference_input, verbose)
    return res_batch[0]


def pre_process_recording_generic(recording, verbose=True):
    '''
    Generic pre-processing to the correct input format before inference for recordings
    :param recording:
    :param verbose:
    :return:
    '''
    inference_input = recording.copy()
    if len(recording.shape) != 4:
        raise ValueError(
            'Expected recording to have shape (nb_frames, channels, height, width), got: ' + str(recording.shape) +
            '.\n Please make sure to provide the recording in the correct format before running inference_recording.')
    if recording.shape[1] != 1:
        if verbose:
            warnings.warn(
                'Warning...........: Expected recording to have 1 grayscale channel, got: ' + str(recording.shape[1]) +
                'channels.\n Only the first channel will be used for inference.')
        inference_input = inference_input[:, 0, :, :]
    nb_frames = inference_input.shape[0]
    if inference_input.shape[2] != 256 or inference_input.shape[3] != 256:
        if verbose:
            print('The given recording has shape: ' + str(inference_input.shape) +
                  '.\n The data is resized to (nb_frames, 1, 256, 256) before running inference.')
            print("Resizing data...")
            # Resize via iteration in order to be able to show progress
            # Initialize an empty list to store resized images
            resized_recording = []
            # Iterate through the images and resize them
            for image in tqdm(inference_input, desc='Resizing images', unit='image'):
                resized_image = resize(image, (inference_input.shape[1], 256, 256), preserve_range=True)
                resized_recording.append(resized_image)
            inference_input = np.array(resized_recording)
        else:
            inference_input = resize(inference_input, (nb_frames, inference_input.shape[1], 256, 256),
                                     preserve_range=True)
        if verbose:
            print("Resizing done.")
    # convert data to float32
    inference_input = inference_input.astype(np.float32)
    return inference_input


def inference_recording_generic(model_object, inference_input, verbose=True):
    '''
    Perform inference on the given recording using the given onnx session.
    :param model_object: ONNXModelObject
        The model object containing the onnx session
        Only onnx is supported
    :param inference_input: np.array
        Pre-processed input data as np.array of shape (nb_frames, channels, height, width)
    :param verbose:  bool
        If True, info and warnings concerning resizing and channels will be printed
    :return: np.array
        Inference output as np.array with either numerical or string labels
        The output of the inference is a ndarray with size nb_framesx8 with the quality labels in the following order:
        basal_left,mid_left,apical_left,apical_right,mid_right,basal_right,annulus_left,annulus_right
    '''
    nb_frames = inference_input.shape[0]
    res = []
    i = 0
    if verbose:
        print("Running inference...")
        progress_bar = tqdm(total=nb_frames)
    else:
        progress_bar = None
    while i < inference_input.shape[0]:
        res.append(model_object.inference_batch(inference_input[i:i + model_object.batch_size]))
        i += model_object.batch_size
        if verbose:
            progress_bar.update(model_object.batch_size)
    if verbose:
        progress_bar.close()
    return np.concatenate(res)


def load_model(model_name='mobilenetv2_regional_quality', **kwargs):
    '''
    Load the model with the given backbone
    :param model_name: str
        The name of the model to load
    :return: ModelObjeect
        The model object containing the onnx session and optionally a slope_intercept file
    '''
    if 'pixel_based' in model_name:
        model_object = arqee.set_up_pixel_based_method(model_name)
    else:
        model_dir = os.path.join(CONST.MODEL_DIR, model_name)
        if not os.path.exists(model_dir):
            raise ValueError('Model directory does not exist: ' + model_dir
                             + '\nPlease download and set up the model using download_model and set_up_model')
        else:
            model_object = arqee.load_onnx_model_from_dir(model_dir, **kwargs)
    return model_object
