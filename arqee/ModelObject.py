import arqee
import numpy as np
from tqdm import tqdm
import os
import onnxruntime
from arqee import utils
import arqee.CONST as CONST
from skimage.transform import resize


class ModelObject:
    def __init__(self, session, batch_size=32):
        '''
        Generic model object for onnx models
        :param session: onnxruntime.InferenceSession
            The onnx session
        :param batch_size: int
            The batch size to use for inference on recordings
        '''
        self.session = session
        self.batch_size = batch_size

    def pre_process_batch(self, inference_input, verbose=True, **kwargs):
        '''
        Pre-process the input before inference
        :param verbose: bool
        :param inference_input: np.array
        :return: np.array
        '''
        return arqee.pre_process_batch_generic(inference_input, verbose=verbose)

    def pre_process_img(self, img_data, verbose=True, **kwargs):
        '''
        Pre-process the input before inference
        :param verbose: bool
        :param img_data: np.array
        :return: np.array
        '''
        return arqee.pre_process_img_generic(img_data, verbose=verbose)

    def pre_process_recording(self, recording_data, verbose=True, **kwargs):
        '''
        Pre-process the input before inference
        :param verbose: bool
        :param recording_data: np.array
        :return: np.array
        '''
        return arqee.pre_process_recording_generic(recording_data, verbose=verbose)

    def post_process_batch(self, inference_output, verbose=True, **kwargs):
        '''
        Perform post processing on the inference output for batches
        :param verbose: bool
        :param inference_output: np.array
        :return: np.array
        '''
        return inference_output

    def post_process_img(self, inference_output, verbose=True, **kwargs):
        '''
        Perform post processing on the inference output for images
        :param verbose: bool
        :param inference_output: np.array
        :return: np.array
        '''
        inference_output_as_batch = np.expand_dims(inference_output, axis=0)
        return self.post_process_batch(inference_output_as_batch, verbose=verbose, **kwargs)[0]

    def post_process_recording(self, inference_output, verbose=True, **kwargs):
        '''
        Perform post processing on the inference output for recordings
        :param verbose: bool
        :param inference_output: np.array
        :return: np.array
        '''
        nb_frames = inference_output.shape[0]
        # convert inference output to a list
        new_inference_output = inference_output.tolist()
        frames_processed = 0
        if verbose:
            print("Post-processing recording inference output...")
            progress_bar = tqdm(total=nb_frames)
        else:
            progress_bar = None
        while frames_processed < nb_frames:
            batch_of_frames = inference_output[frames_processed:frames_processed + self.batch_size]
            post_processed_batch = self.post_process_batch(batch_of_frames, verbose=verbose, **kwargs)
            new_inference_output[frames_processed:frames_processed + self.batch_size] = post_processed_batch
            frames_processed += self.batch_size
            if verbose:
                progress_bar.update(self.batch_size)
        if verbose:
            progress_bar.close()
        return np.array(new_inference_output)

    def inference_batch(self, batch_data, verbose=True, **kwargs):
        '''
        Perform inference on the given batch of data
        :param verbose: bool
        :param batch_data: np.array
        :return: np.array
        '''
        return arqee.inference_batch_generic(self, batch_data, verbose=verbose)

    def inference_img(self, img_data, verbose=True, **kwargs):
        '''
        Perform inference on the given image
        :param verbose: bool
        :param img_data: np.array
        :return: np.array
        '''
        return arqee.inference_img_generic(self, img_data, verbose=verbose)

    def inference_recording(self, recording_data, verbose=True, **kwargs):
        '''
        Perform inference on the given recording
        :param verbose: bool
        :param recording_data: np.array
        :return: np.array
        '''
        return arqee.inference_recording_generic(self, recording_data, verbose=verbose)

    def predict_img(self, img_data, verbose=True, **kwargs):
        '''
        Perform inference on the given image using
        :param verbose: bool
        :param img_data: np.array
        :return: np.array
        '''
        inference_input = self.pre_process_img(img_data, verbose=verbose, **kwargs)
        inference_output = self.inference_img(inference_input, verbose=verbose, **kwargs)
        return self.post_process_img(inference_output, verbose=verbose, **kwargs)

    def predict_batch(self, batch_data, verbose=True, **kwargs):
        '''
        Perform inference on the given batch of data using
        :param verbose: bool
        :param batch_data: np.array
        :return: np.array
        '''
        inference_input = self.pre_process_batch(batch_data, verbose=verbose, **kwargs)
        inference_output = self.inference_batch(inference_input, verbose=verbose, **kwargs)
        return self.post_process_batch(inference_output, verbose=verbose, **kwargs)

    def predict_recording(self, recording_data, verbose=True, **kwargs):
        '''
        Perform inference on the given recording
        :param recording_data: np.array
        :return: np.array
        '''
        inference_input = self.pre_process_recording(recording_data, verbose=verbose, **kwargs)
        inference_output = self.inference_recording(inference_input, verbose=verbose, **kwargs)
        return self.post_process_recording(inference_output, verbose=verbose, **kwargs)


class QualityModelObject(ModelObject):
    def __init__(self, session, slope_intercept_bias_correction, batch_size=32):
        '''
        Model object containing the onnx session and slope_intercept for bias correction
        :param session: onnxruntime.InferenceSession
            The onnx session
        :param batch_size: int
            The batch size to use for inference on recordings
        :param slope_intercept_bias_correction: np.array
            Array of length 2 containing (slope, intercept) as floats
            The slope is the slope of the linear model fitted to the validation output and labels
            The intercept is the intercept of the linear model fitted to the validation output and labels
            These values are used to correct for bias in the regression model from underfitting and regression dilution
        '''
        super().__init__(session, batch_size=batch_size)
        self.slope_intercept_bias_correction = slope_intercept_bias_correction

    def post_process_batch(self, inference_output, verbose=True, **kwargs):
        '''
        Quality model specific post-processing.
        - Correct for bias using the slope_intercept if provided.
            The slope and intercept are obtained by fitting a linear model to the
            validation output and labels. When doing inference, we then use these values to correct for the bias which
            is introduced due to underfitting and regression dilution.
            This is inspired by:
            Peng, Han, et al. "Accurate brain age prediction with lightweight deep neural networks." Medical image analysis 68 (2021): 101871.
            https://www.sciencedirect.com/science/article/pii/S1361841520302358
        - Optionally convert the numerical output to string labels, if kwargs['convert_to_labels'] is True
        :param verbose: bool
        :param inference_output: np.array
            Inference output as np.array of size (batch_size,nb_labels)
        :return: np.array
            Post processed inference output with size (batch_size,nb_labels)
        '''
        apply_bias_correction = kwargs.get('apply_bias_correction', True)
        if self.slope_intercept_bias_correction is not None and apply_bias_correction:
            inference_output = arqee.apply_linear_model(inference_output,
                                                        self.slope_intercept_bias_correction)
        convert_to_categorical = kwargs.get('convert_to_labels', False)
        if convert_to_categorical:
            numerical_to_labels_vectorized = np.vectorize(arqee.numerical_to_categorical)
            inference_output = numerical_to_labels_vectorized(inference_output)
        return inference_output


class SegmentationModelObject(ModelObject):
    def __init__(self, session):
        '''
        Model object containing the onnx session
        :param session: onnxruntime.InferenceSession
            The onnx session
        '''
        super().__init__(session)

    def post_process_batch(self, inference_output, verbose=True, **kwargs):
        '''
        Segmentation model specific post processing: convert model output to categorical integer labels by
        taking the argmax of the output along the channel axis
        :param verbose: bool
        :param inference_output: np.array
            Inference output as np.array with shape (batch_size, nb_classes, width, height)
        :return: np.array
            Post processed inference output as np.array with shape (batch_size, width, height)
        '''
        return np.argmax(inference_output, axis=1).astype(np.uint8)

class GCNSegmentationModelObject(ModelObject):
    def __init__(self, session,**kwargs):
        '''
        Model object containing the onnx session
        :param session: onnxruntime.InferenceSession
            The onnx session
        :param kwargs: dict
            Optional arguments:
            - output_keypoints: bool
                If True, the keypoints will be not be converted to a segmentation mask
        '''
        super().__init__(session)
        if 'output_keypoints' in kwargs:
            self.output_keypoints = kwargs['output_keypoints']
        else:
            self.output_keypoints = False

    def pre_process_batch(self, inference_input, verbose=True, **kwargs):
        '''
        Pre-processing for GCN to the correct input format before inference for batches.
        Not using the generic pre_process_batch method because the GCN uses 3 channels as input
        :param inference_input: np.array
            Input data as np.array of shape (batch_size, channels, height, width
        :param verbose: bool
            If True, warnings will be printed concerning resizing and channels
        :return: np.array
            Pre-processed input data as np.array of shape (batch_size, 3, 256, 256)
        '''
        # check if data has correct shape
        inference_input = inference_input.copy()
        if len(inference_input.shape) != 4:
            raise ValueError(
                'Expected data to have shape (batch_size, channels, 256, 256), got: ' + str(inference_input.shape) +
                '.\n Please make sure to provide the data in the correct format before running inference_batch.')
        if inference_input.shape[1] != 3:
            if inference_input.shape[1] ==1:
                print('The given data has shape: ' + str(inference_input.shape) +
                      '.\n The data will be resized to (batch_size, 3, 256, 256) before running inference.'
                      'The GCN uses 3 channels as input.')
                # copy the grayscale channel 3 times to create a 3 channel image
                inference_input = np.stack((inference_input[:,0,:,:],inference_input[:,0,:,:],inference_input[:,0,:,:]),axis=1)
            else:
                raise ValueError(
                    'Expected data to have either 3 or 1 channel, got: ' + str(inference_input.shape[1]) +
                    'channels.\n Please make sure to provide the data in the correct format before running inference_batch.')
        if inference_input.shape[2] != 256 or inference_input.shape[3] != 256:
            if verbose:
                print('The given data has shape: ' + str(inference_input.shape) +
                      '.\n The data will be resized to (batch_size, 1, 256, 256) before running inference.')
            inference_input = resize(inference_input,
                                     (inference_input.shape[0], inference_input.shape[1], 256, 256),
                                     preserve_range=True)
        # convert data to float32
        inference_input = inference_input.astype(np.float32)
        return inference_input

    def pre_process_img(self, img_data, verbose=True, **kwargs):
        '''
        Pre-processing to the correct input format before inference for single images.
        Not using the generic pre_process_img method because the GCN uses 3 channels as input
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
        return self.pre_process_batch(inference_input, verbose)

    def inference_img(self, img_data, verbose=True, **kwargs):
        '''
        Perform inference on the given image using the given onnx session.
        Not using the generic inference_img method because the GCN has multiple outputs
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
        onnx_sess = self.session
        input_name = onnx_sess.get_inputs()[0].name
        output_names = [output.name for output in onnx_sess.get_outputs()]
        res = onnx_sess.run(output_names, {input_name: img_data})
        return res

    def post_process_img(self, inference_output, verbose=True, **kwargs):
        '''
        Perform post processing on the inference output for images.
        Not using the generic post_process_img method because the GCN has multiple outputs and thus
        we work with lists instead of np.arrays to avoid issues with different shapes
        :param verbose: bool
        :param inference_output: np.array
        :return: np.array
        '''
        # add batch dimension.
        inference_output_as_batch = [inference_output]
        return self.post_process_batch(inference_output_as_batch, verbose=verbose, **kwargs)[0]


    def post_process_batch(self, inference_output, verbose=True, **kwargs):
        '''
        Graph segmentation model specific post processing: convert keypoints to segmentation mask
        :param verbose: bool
        :param inference_output: np.array
        :return: np.array
            Post processed inference output as np.array with shape (batch_size, width, height)
        '''
        if self.output_keypoints:
            return inference_output
        else:
            # TODO
            raise NotImplementedError('Only output_keypoints=True is supported for GCN segmentation models at the moment.')


class PixelBasedModelObject(ModelObject):
    def __init__(self,slope_intercept,metric='gcnr'):
        '''
        Model object containing the slope and intercept to map from quality metrics to quality predictions
        :param slope_intercept: np.array
            Array of length 2 containing (slope, intercept) as floats
            The slope and intercept are the parameters of the linear model used to predict qualities
            from quality metrics
        '''
        super().__init__(None)
        self.slope_intercept = slope_intercept
        self.metric = metric


    def get_pixels_in_mask(self, img_data, mask, eps=1e-6):
        '''
        Get the pixels in the mask inside the sector
        :param img_data: np.array
            The image data as np.array with shape (width, height)
        :param mask: np.array
            The mask as np.array with shape (width, height)
        :param eps: float
            Threshold for pixels inside the sector
        :return: np.array
            The pixels in the mask
        '''
        in_sector_mask = img_data>eps
        combined_mask = np.logical_and(mask, in_sector_mask)
        return img_data[combined_mask>0]

    def get_quality_metric_region(self,
                                  us_image,
                                  roi_mask,
                                  bg_mask=None,
                                  **kwargs):
        '''
        Predict the quality metric of a region with the pixel-based method.
        :param us_image: np.array
            The ultrasound image as np.array with shape (width, height)
        :param roi_mask: np.array
            The mask of the region of interest as np.array with shape (width, height)
        :param bg_mask: np.array
            The mask of the region as np.array with shape (width, height)
            This mask is only needed if the metric needs a background region
        :param kwargs: dict
            Optional arguments:
            - eps: float
                Threshold for the region mask
            - apply_linear_model: bool
                If True, apply the linear model to the quality metric, otherwise return the raw quality metric
        :return: float
            The quality metric of the region
        '''
        eps = kwargs.get('eps',1e-6)
        roi_pixels = self.get_pixels_in_mask(us_image, roi_mask, eps=eps)
        quality_metric=0.0
        if self.metric == 'intensity':
            quality_metric = np.mean(roi_pixels)
        elif self.metric in ['cr','cnr','gcnr']:
            # these metrics require a background mask
            if bg_mask is None:
                raise ValueError(f'Background mask is required for metric {self.metric}')
            bg_pixels = self.get_pixels_in_mask(us_image, bg_mask, eps=eps)
            if self.metric == 'cr':
                quality_metric = np.mean(roi_pixels)/np.mean(bg_pixels)
            elif self.metric == 'cnr':
                quality_metric = utils.contrast_to_noise_ratio(roi_pixels,bg_pixels)
            elif self.metric == 'gcnr':
                quality_metric = utils.gcnr(roi_pixels,bg_pixels)
        else:
            raise ValueError(f'Unknown metric: {self.metric}')
        apply_linear_model = kwargs.get('apply_linear_model', True)
        if self.slope_intercept is not None and apply_linear_model:
            quality_metric = arqee.apply_linear_model(quality_metric, self.slope_intercept)
        convert_to_categorical = kwargs.get('convert_to_labels', False)
        if convert_to_categorical:
            numerical_to_labels_vectorized = np.vectorize(arqee.numerical_to_categorical)
            quality_metric = numerical_to_labels_vectorized(quality_metric)
        return quality_metric


    def inference_img(self, img_data, verbose=True,**kwargs):
        normalized_img = utils.normalize_image(img_data[0][0])
        seg = kwargs.get('segmentation',None)
        if seg is None:
            raise ValueError(f'Segmentation missing. Segmentation must be provided '
                             f'as a keyword argument (segmentation=...) for pixel-based methods')
        divided_seg = arqee.divide_segmentation(include_lv_lumen=True,
                                                **kwargs) # not passing segmentation as it is
                                                          # already passed as a keyword argument in kwargs
        quality_metrics = []
        if self.metric in ['cr', 'cnr', 'gcnr']:
            bg_mask = divided_seg[-1]
        else:
            bg_mask = None
        for region_mask in divided_seg[:-1]:
            quality_metric = self.get_quality_metric_region(normalized_img, region_mask, bg_mask=bg_mask,**kwargs)
            quality_metrics.append(quality_metric)
        return np.array(quality_metrics)



    def inference_batch(self, batch_data, verbose=True, **kwargs):
        '''
        Perform inference on the given batch of data.
        This overwrites the default inference_batch_generic method as we do not have an onnx session.
        :param verbose: bool
        :param batch_data: np.array
        :return: np.array
        '''
        res = []
        segmentations_batch = kwargs.get('segmentation',None)
        if segmentations_batch is None:
            raise ValueError(f'Segmentation missingg. Segmentation must be provided '
                             f'as a keyword argument (segmentation=...) for pixel-based methods')
        for img,seg in zip(batch_data,segmentations_batch):
            # expand dims of img to (1,channels,width,height)
            img = np.expand_dims(np.expand_dims(img, axis=0))
            res.append(self.inference_img(img,segmentation=seg,verbose=verbose,**kwargs))
        return np.array(res)


def load_onnx_model_from_dir(model_dir, **kwargs):
    '''
    Load the onnx model from the given model directory. Also load the slope and intercept for bias correction if
    available.
    :param model_dir:
    The model_dir should have the following structure:
        model_dir
            model.onnx
            Extra optional files:
            - for quality model: slope_intercept_bias_correction.npy
                slope and intercept of the linear model used to correct for bias in the regression model saved
                as a numpy array of form (slope, intercept)
        Where model.onnx is the onnx model and slope_intercept.npy is a numpy array containing the slope and intercept
        for bias correction.
    :return: onnxruntime.InferenceSession, np.array
        (session, slope_intercept)
        where
        - session is the onnxruntime.InferenceSession object
        - slope_intercept is a numpy array containing (slope, intercept) for bias correction as floats
    '''
    if not os.path.exists(model_dir):
        raise ValueError('Model directory does not exist: ' + model_dir)
    if not os.path.exists(os.path.join(model_dir, 'model.onnx')):
        raise ValueError('No ONNX model found in model directory: ' + model_dir)
    onnx_model_loc = os.path.join(model_dir, 'model.onnx')

    sess = onnxruntime.InferenceSession(onnx_model_loc, None)
    model_name = os.path.basename(model_dir)
    if 'regional_quality' in model_name:
        slope_intercept_bias_correction_loc = os.path.join(model_dir, 'slope_intercept_bias_correction.npy')
        if os.path.exists(slope_intercept_bias_correction_loc):
            slope_intercept_bias_correction = np.load(slope_intercept_bias_correction_loc)
        else:
            slope_intercept_bias_correction = None
        model_object = QualityModelObject(sess, slope_intercept_bias_correction)
    elif 'unet' in model_name:
        model_object = SegmentationModelObject(sess)
    elif 'GCN' in model_name:
        model_object = GCNSegmentationModelObject(sess, **kwargs)
    else:
        model_object = ModelObject(sess)
    return model_object


def set_up_pixel_based_method(quality_metric,slope_intercept=None):
    if 'gcnr' in quality_metric:
        metric = 'gcnr'
        if slope_intercept is None:
            slope_intercept = CONST.DEFAULT_SLOPE_INTERCEPTS_PIXEL_BASED_METHODS['gcnr']
    elif 'cnr' in quality_metric:
        metric = 'cnr'
        if slope_intercept is None:
            slope_intercept = CONST.DEFAULT_SLOPE_INTERCEPTS_PIXEL_BASED_METHODS['cnr']
    elif 'cr' in quality_metric:
        metric = 'cr'
        if slope_intercept is None:
            slope_intercept = CONST.DEFAULT_SLOPE_INTERCEPTS_PIXEL_BASED_METHODS['cr']
    elif 'intensity' in quality_metric:
        metric = 'intensity'
        if slope_intercept is None:
            slope_intercept = CONST.DEFAULT_SLOPE_INTERCEPTS_PIXEL_BASED_METHODS['intensity']
    else:
        raise ValueError(f'Unknown quality metric: {quality_metric}, supported metrics are gcnr, cnr, cr, '
                         f'and intensity')
    return PixelBasedModelObject(slope_intercept,metric)









