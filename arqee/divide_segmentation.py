import numpy as np
from skimage.measure import find_contours
import cv2


def line(p1, p2, img_width=256, img_height=256, cont=True):
    '''
    Bresenham's line algorithm
    :param p1: ndarray
        numpy array of shape (2,) containing the coordinates of the start point
    :param p2: ndarray
        numpy array of shape (2,) containing the coordinates of the end point
    :param img_width: int, optional
        width of the image
    :param img_height: int, optional
        height of the image
    :param cont: bool, optional
        whether to draw a continuous line or limit the line to the points between the start and end points
    :return: ndarray
        numpy array of shape (n,2) containing the coordinates of the line
    '''
    x0, y0 = p1
    x1, y1 = p2
    steep = abs(y1 - y0) > abs(x1 - x0)  # abs(r) > 0
    width = img_width
    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        width = img_height
    flip = False
    if x0 > x1:
        x0, y0, x1, y1 = x1, y1, x0, y0
        flip = True
    deltax = x1 - x0
    deltay = abs(y1 - y0)
    error = -deltax / 2
    if cont:
        if x1 - x0 != 0:
            slope = (y1 - y0) / (x1 - x0)
        else:
            slope = np.max(img_width, img_height)
        y = y0 - slope * x0
        x0 = 0
    else:
        y = y0
    if y < y1:
        ystep = 1
    else:
        ystep = -1
    line = []
    if cont:
        it_range = range(0, width)
    else:
        it_range = range(x0, x1 + 1)
    for x in it_range:
        if steep:
            if 0 <= y < img_width:  # check out of bounds
                line.append((int(y), int(x)))
        else:
            if 0 <= y < img_height:  # check out of bounds
                line.append((int(x), int(y)))
        error = error + deltay
        if error > 0:
            y = y + ystep
            error = error - deltax
    if flip:
        line = np.flip(line, axis=0)
    return line


def get_closest_point(point, contour):
    '''
    Get the index and the coordinates of the closest point in a given contour to a given point
    :param point: ndarray
        numpy array of shape (2,) containing the coordinates of the point
    :param contour: ndarray
        numpy array of shape (n,2) containing the coordinates of the contour
    :return: tuple
        tuple containing the index and the coordinates of the closest point in the contour
        If the contour is empty, returns None,np.inf
    '''
    min_distance = np.inf
    closest_point = None
    for i, contour_point in enumerate(contour):
        euclidean_distance = np.linalg.norm(point - contour_point)
        if euclidean_distance < min_distance:
            closest_index = i
            closest_point = contour_point
            min_distance = euclidean_distance
    return closest_index, closest_point


def get_region_contour(start_index, end_index, myo_contour, contour_lv,
                       myo_start_index=None, myo_point_start=None,
                       myo_end_index=None, myo_point_end=None,
                       img_width=256, img_height=256):
    '''
    Get the contour of the region defined by the given start and end indices on the endocardium border.
    The region contour consists of four parts:
     1) the first part is the part along the endocardium border
     2) the second part is a straight line from the end point on the endocardium border to the start point on the
        epicardium border
     3) the third part is the part on the epicardium border
     4) the fourth part is a straight line from the end point on the epicardium border to the start point on the
        endocardium border, closing the region
    :param start_index: int
        index of the start point on the endocardium border
    :param end_index: int
        index of the end point on the endocardium border
    :param myo_contour: ndarray
        numpy array of shape (n,2) containing the coordinates of the myocardium contour in counterclockwise order
        starting near the apex
    :param contour_lv: ndarray
        numpy array of shape (n,2) containing the coordinates of the left ventricle boundary in counterclockwise order
        starting near the apex
    :param myo_start_index: int, optional
        index of point on the myocardium contour corresponding to the start endocardium point
    :param myo_point_start: ndarray, optional
        coordinates of point on the myocardium contour corresponding to the start endocardium point
    :param myo_end_index: int, optional
        index of point on the myocardium contour corresponding to the end endocardium point
    :param myo_point_end: ndarray, optional
        coordinates of point on the myocardium contour corresponding to the end endocardium point
    :param img_width: int, optional
        width of the image, default is 256
    :param img_height: int, optional
        height of the image, default is 256
    :return: tuple
        returns (division_contour,myo_end_index,myo_point_end), where
        - division_contour: ndarray
            numpy array of shape (n,2) containing the coordinates of the region contour points in counterclockwise order
        - myo_end_index: int
            index of the end point on the myocardium contour
        - myo_point_end: ndarray
            coordinates of the end point on the myocardium contour
        The last two return values can be used to avoid re-computation when dividing successive regions
    '''
    # 1) the first part is the part along the endocardium border
    lv_start_point = contour_lv[start_index].astype(int)
    lv_boundary_part = []
    # contour_lv is in counterclockwise order starting near the apex, so in order to get the region contour in
    # counterclockwise order, we iterate over in reverse order as left and right is flipped
    lv_index = end_index
    while lv_index != start_index:
        lv_boundary_part.append(contour_lv[lv_index])
        lv_index -= 1
        if lv_index == -1:
            lv_index = len(contour_lv) - 1
    division_contour = lv_boundary_part
    # 2) the second part is a straight line from the end point on the endocardium border to the start point on the
    #    epicardium border
    # get the corresponding point of the start endocardium point on the myocardium contour if not given
    if myo_start_index is None or myo_point_start is None:
        myo_start_index, myo_point_start = get_closest_point(lv_start_point, myo_contour)
    division_contour = np.append(division_contour, np.array(line(lv_start_point,
                                                                 myo_point_start,
                                                                 cont=False,
                                                                 img_width=img_width,
                                                                 img_height=img_height)), axis=0)

    # 3) the third part is the part on the epicardium border
    lv_point_end = contour_lv[end_index].astype(int)
    # get the corresponding point of the end endocardium point on the myocardium contour if not given
    if myo_end_index is None or myo_point_end is None:
        myo_end_index, myo_point_end = get_closest_point(lv_point_end, myo_contour)
    # get the part of the myocardium contour between the start and end points. Now, we can iterate in normal order
    myo_index = myo_start_index
    while myo_index != myo_end_index:
        division_contour = np.append(division_contour, np.array([myo_contour[myo_index]]), axis=0)
        myo_index += 1
        if myo_index == len(myo_contour):
            myo_index = 0
    # 4) the fourth part is a straight line from the end point on the epicardium border to the start point on the
    #    endocardium border, closing the region
    division_contour = np.append(division_contour, line(myo_point_end,
                                                        contour_lv[end_index].astype(int),
                                                        cont=False,
                                                        img_width=img_width,
                                                        img_height=img_height),
                                 axis=0)
    return division_contour, myo_end_index, myo_point_end


def find_apex(contour, base_mid):
    '''
    Find the apex point on the endocardium.
    More precisely, the apex is defined as the point the furthest from the midpoint
    :param contour: ndarray
        numpy array of shape (n,2) containing the coordinates of the left ventricle boundary or myocardium contour
    :param base_mid: ndarray
        numpy array of shape (2,) containing the coordinates of middle of the base
    :return: tuple
        returns (apex,apex_index), where
        - apex: ndarray
            numpy array of shape (2,) containing the coordinates of the apex point
        - apex_index: int
            index of the apex in the LV_boundary
    '''
    # Find apex point on the endocardium, as point with highest distance from baseMid
    distances = []
    for point in contour:
        distances.append((np.linalg.norm(base_mid - point)))
    # Get point with highest distance from baseMid
    apex_index = np.argmax(distances)
    apex = np.copy(contour[apex_index])
    return apex, apex_index


def find_epi_apex(myo_mask, baseMid, lv_apex):
    '''
    Find the epicardial apex point, defined as the last point on the line through the baseMid and the lv_apex still
    inside the myocardium mask.
    :param myo_mask: ndarray
        numpy array of shape (height,width) containing the myocardium mask
    :param baseMid:  ndarray
        numpy array of shape (2,) containing the coordinates of middle of the base
    :param lv_apex:  ndarray
        numpy array of shape (2,) containing the coordinates of the apex point on the endocardium
    :return:  ndarray
        numpy array of shape (2,) containing the coordinates of the epicardial apex point
    '''
    # Find apex point on the epicardium, as intersection of the long axis (Apex-BaseMid) and the epicardial border
    epi_apex = lv_apex.copy()
    # Get the direction of the line
    direction = np.array([lv_apex[0] - baseMid[0], lv_apex[1] - baseMid[1]])
    # normalize the direction
    direction = direction / direction[0]
    for depth in range(0, lv_apex[1]):
        point = (lv_apex + direction * depth).astype(int)
        if myo_mask[point[1], point[0]] > 0.5:
            epi_apex = point
        else:
            break

    return epi_apex


def find_lv_landmarks(segmentation, contour_lv, contour_myo, la_label, ao_label, myo_label):
    '''
    Find the base points and the apex of the left ventricle in a given segmentation
    :param segmentation: ndarray
        numpy array of shape (height,width) containing segmentation data as labelled array (not one-hot)
    :param contour_lv: ndarray
        numpy array of shape (n,2) containing the coordinates of the left ventricle boundary in counterclockwise order
        starting near the apex
    :param contour_myo: ndarray
        numpy array of shape (n,2) containing the coordinates of the myocardium contour in counterclockwise order
        starting near the apex
    :param la_label: int
        label of the left atrium in the segmentation
    :param ao_label: int
        label of the aorta in the segmentation
    :return: tuple
         return (landmarks,landmark_indices),
         where
        - landmarks = (base_sep,base_lat,apex,epi_apex)
        - landmark_indices = (base_sep_index,base_lat_index,apex_index,epi_apex_index)
        where
        - base_sep: ndarray
            numpy array of shape (2,) containing the coordinates of the base point on the septal side
        - base_lat: ndarray
            numpy array of shape (2,) containing the coordinates of the base point on the left side
        - apex: ndarray
            numpy array of shape (2,) containing the coordinates of the apex point
        - epi_apex: ndarray
            numpy array of shape (2,) containing the coordinates of the apex point on the myocardium
        - base_sep_index: int
            index of the base point on the septal side in the contour_lv
        - base_lat_index: int
            index of the base point on the left side in the contour_lv
        - apex_index: int
            index of the apex in the contour_lv
        - myo_contour: ndarray
            numpy array of shape (n,2) containing the coordinates of the myocardium contour
    '''
    # Iterate over lv contour and classify each contour point based on its surroundings
    # base_boundary is the boundary between the lv and the la or aorta
    # la_boundary is the boundary between the lv and la. It is a subset of base_boundary
    # for A2C/A4C, base_boundary == la_boundary
    # base_indices are the indices of the base_boundary points in contour_lv
    base_boundary = []
    la_boundary = []
    base_indices = []
    index = 0
    for point in contour_lv:
        x = int(round(point[0]))
        y = int(round(point[1]))
        # Check if atrium is below
        for offset in range(1, 5, 1):
            ind = y + offset
            if ind >= segmentation.shape[0]:
                # If we reach the bottom of the image, the atrium or aorta is outside the image
                base_boundary.append(np.array([x, y]))
                base_indices.append(index)
                break
            else:
                pixel_label = segmentation[y + offset, x]
                condition = pixel_label == la_label or pixel_label == ao_label
                if condition:
                    if pixel_label == la_label:
                        la_boundary.append(np.array([x, y]))
                    # Atrium
                    base_boundary.append(np.array([x, y]))
                    base_indices.append(index)
                    break
        index += 1
    # The septal base point is the first point in base_boundary because the contour is clockwise and starts near the apex
    # The lateral base point is the last point in base_boundary because the contour is clockwise and starts near the apex
    base_sep = base_boundary[0]
    base_lat = base_boundary[-1]
    base_sep_index = base_indices[0]
    base_lat_index = base_indices[-1]
    # Get the apex, defined as the lv point furthest away from the mid of the la base point
    base_mid_la = la_boundary[int(len(la_boundary) / 2)]
    apex, apex_index = find_apex(contour_lv, base_mid_la)
    myo_mask = segmentation == myo_label
    epi_apex = find_epi_apex(myo_mask, base_mid_la, apex)
    epi_apex_index, epi_apex = get_closest_point(epi_apex, contour_myo)
    landmarks = (base_sep, base_lat, apex, epi_apex)
    landmark_indices = (base_sep_index, base_lat_index, apex_index, epi_apex_index)
    return (landmarks, landmark_indices)


def flip_y(arr, max_val=255, y_ind=0):
    '''
    Flip the y-coordinate of a given contour
    :param arr: ndarray
        numpy array of shape (n,2) containing the coordinates of the contour
    :param max_val: int, optional
        maximum value of the y-coordinate
    :param y_ind: int, optional
        index of the y-coordinate in the contour
    :return: ndarray
        numpy array of shape (n,2) containing the coordinates of the flipped contour
    '''
    if len(arr.shape) == 1:
        arr[y_ind] = max_val - arr[y_ind]
    else:
        for i in range(len(arr)):
            arr[i][y_ind] = max_val - arr[i][y_ind]
    return arr


def get_contour(segmentation, labels, start_near_top=True):
    '''
    Get the contour of the structure with the given label in the given segmentation
    :param segmentation: ndarray
        numpy array of shape (height,width) containing segmentation data as labelled array (not one-hot)
    :param labels: int|tuple
        either a single label or a tuple of labels of structures in the segmentation to get the contour of
        if a tuple, the contours of the structures will be combined
    :param start_near_top: bool, optional
        whether to start the contour near the top. If True, the contour will be in counterclockwise order
        starting near the apex.
        If False, the contour will be in clockwise order starting at the bottom.
    :return: ndarray
        numpy array of shape (n,2) containing the coordinates of the contour
    '''
    if isinstance(labels, int):
        labels = (labels,)
    if start_near_top:
        flipped_segmentation = np.copy(segmentation)
        flipped_segmentation = np.flipud(flipped_segmentation)
        mask = np.isin(flipped_segmentation, labels)
    else:
        mask = np.isin(segmentation, labels)
    # find_contours gives us the contour in clockwise order starting at the bottom.
    # Because of the flip, the contour is in counterclockwise order starting near the apex, if flipped
    contours = find_contours(mask, level=0.5)
    lengths = [len(cont) for cont in contours]
    largest_contour = contours[np.argmax(lengths)]
    if start_near_top:
        # Flip back to original orientation
        largest_contour = flip_y(largest_contour, y_ind=0, max_val=segmentation.shape[0] - 1)
    # reverse columns to get (x,y) instead of (y,x)
    contour = np.round(np.flip(largest_contour, axis=1)).astype(int)
    return contour


def divide_segmentation(segmentation,**kwargs):
    '''
    Divide the myocardium into six regions: 3 left and 3 right regions.
    First the endocardium is divided into left and right parts, based on the location of the apex.
    Then, both the left and the right encocardium parts are divided into three equal parts.
    The contours of the regions are then obtained by finding the closest myocardium points corresponding to the start
    and end endocardium points and closing the region with two straight lines between the endocardium and the myocardium.
    Optionally, the two annulus points can be included as regions. These are defined as circles around the septal and
    lateral base points with the given radius.
    :param segmentation: ndarray
        numpy array of shape (labels,height,width) containing the segmentation (one-hot encoded) or
        ndarray of shape (height,width) containing the segmentation (not one-hot encoded)
    :param kwargs: dict, optional
        dictionary containing the optional arguments:
        - myo_label: int,
            label of the myocardium in the segmentation
            default value is 2
        - lv_label: int
            label of the left ventricle in the segmentation
            default value is 1
        - la_label: int
            label of the left atrium in the segmentation
            default value is 3
        - ao_label: int
            label of the aorta in the segmentation
            default value is 4
        - new_labels: list
            list of integers containing the new labels to use for each of the regions, in the order:
            [top_left,middle_left,bottom_left,bottom_right,middle_right,top_right,annulus_left,annulus_right]
            default value is [0, 1, 2, 3, 4, 5, 6, 7]
        - include_annulus: bool
            whether to include the annulus points as regions
            default value is True
        - annulus_area_radius: int
            radius of the annulus area in pixels
            default value is 10
        - include_lv_lumen: bool
            whether to include the left ventricle lumen in the segmentation as a region.
            The mask is appended to the segmentation.
            default value is False
    :return: ndarray
        numpy array of shape (n_classes,height,width) containing a new one-hot-labelled segmentation with the regions.
        The regions are in the order:
        [bottom_left,middle_left,top_left,top_right,middle_right,bottom_right,annulus_left,annulus_right], where
        the two last regions are only included if include_annulus is True, i.e., n_classes = 6 if include_annulus is
        False and n_classes = 8 if include_annulus is True.
        Addtionally, if include_lv_lumen is True, the left ventricle lumen is included, so n_classes = 7 or 9 with
        and without annulus, respectively.
    '''
    new_labels=kwargs.get('new_labels', [0, 1, 2, 3, 4, 5, 6, 7])
    myo_label=kwargs.get('myo_label', 2)
    lv_label=kwargs.get('lv_label', 1)
    la_label=kwargs.get('la_label', 3)
    ao_label=kwargs.get('ao_label', 4)
    include_annulus=kwargs.get('include_annulus', True)
    annulus_area_radius=kwargs.get('annulus_area_radius', 10)
    include_lv_lumen=kwargs.get('include_lv_lumen', False)
    if len(segmentation.shape) == 3:
        # convert one-hot to categorical
        segmentation = np.argmax(segmentation, axis=0).astype(np.uint8)

    # get image dimensions
    height, width = segmentation.shape

    if segmentation.dtype != np.uint8:
        raise ValueError('Input segmentation has to be in categorial format')

    # STEP 0 : get contours
    # Get the contours of the left ventricle in counterclockwise order starting near the apex
    contour_lv = get_contour(segmentation, lv_label, start_near_top=True)
    # Get the outer contour of the myocardium in counterclockwise order starting near the apex
    # We get the outer border by merging the lv and myocardium masks
    outer_myo_contour = get_contour(segmentation, (myo_label, lv_label), start_near_top=True)

    # STEP 1: Find landmarks of myocardium
    landmarks, landmark_indices = \
        find_lv_landmarks(segmentation, contour_lv, outer_myo_contour, la_label, ao_label, myo_label)
    (base_sep, base_lat, apex, epi_apex) = landmarks
    (base_sep_index, base_lat_index, apex_index, epi_apex_index) = landmark_indices

    # STEP 2: divide endocardium contour into left and right parts, each splitted into 3 parts
    # the contour starts at the top near the apex, so the apex is either somewhere at the start or somewhere at the end
    apex_at_end = False
    if apex_index > base_sep_index:
        apex_at_end = True
    # left
    if apex_at_end:
        left_indices_part1 = np.arange(apex_index, len(contour_lv))
        left_indices_part2 = np.arange(0, base_sep_index)
        left_indices = np.concatenate((left_indices_part1, left_indices_part2))
    else:
        left_indices = np.arange(apex_index, base_sep_index)
    # divide left part into three equal parts
    divided_left_indices = np.array_split(left_indices, 3)
    # right
    if apex_at_end:
        right_indices = np.arange(base_lat_index, apex_index)
    else:
        right_indices_part1 = np.arange(base_lat_index, len(contour_lv))
        right_indices_part2 = np.arange(0, apex_index)
        right_indices = np.concatenate((right_indices_part1, right_indices_part2))
    # divide right part into three equal parts
    divided_right_indices = np.array_split(right_indices, 3)

    # STEP 3: Get the myocardium region connected to each part
    nb_classes = max(new_labels) + 1
    result = np.zeros((nb_classes, segmentation.shape[0], segmentation.shape[1]), np.uint8)

    # left
    # top
    # get region contour
    left_top_contour, myo_end_index1, myo_point_end1 = (
        get_region_contour(divided_left_indices[0][0], divided_left_indices[0][-1], outer_myo_contour, contour_lv,
                           myo_start_index=epi_apex_index, myo_point_start=epi_apex,
                           img_width=width, img_height=height))
    # fill the mask in the result
    label = new_labels[2]
    cv2.fillPoly(result[label], [left_top_contour.astype(np.int32)], color=(1, 0, 0))
    # middle
    # get region contour
    left_middle_contour, myo_end_index2, myo_point_end2 = (
        get_region_contour(divided_left_indices[0][-1], divided_left_indices[1][-1], outer_myo_contour, contour_lv,
                           myo_start_index=myo_end_index1, myo_point_start=myo_point_end1,
                           img_width=width, img_height=height))
    # fill the mask in the result
    label = new_labels[1]
    cv2.fillPoly(result[label], [left_middle_contour.astype(np.int32)], color=(1, 0, 0))
    # bottom
    # get region contour
    left_bottom_contour, _, _ = (
        get_region_contour(divided_left_indices[1][-1], divided_left_indices[2][-1], outer_myo_contour, contour_lv,
                           myo_start_index=myo_end_index2, myo_point_start=myo_point_end2,
                           img_width=width, img_height=height))
    # fill the mask in the result
    label = new_labels[0]
    cv2.fillPoly(result[label], [left_bottom_contour.astype(np.int32)], color=(1, 0, 0))

    # right
    # bottom
    # get region contour
    right_top_contour, myo_end_index3, myo_point_end3 = (
        get_region_contour(divided_right_indices[0][0], divided_right_indices[0][-1], outer_myo_contour, contour_lv,
                           img_width=width, img_height=height))
    # fill the mask in the result
    label = new_labels[5]
    cv2.fillPoly(result[label], [right_top_contour.astype(np.int32)], color=(1, 0, 0))
    # middle
    # get region contour
    right_middle_contour, myo_end_index4, myo_point_end4 = (
        get_region_contour(divided_right_indices[0][-1], divided_right_indices[1][-1], outer_myo_contour, contour_lv,
                           myo_start_index=myo_end_index3, myo_point_start=myo_point_end3,
                           img_width=width, img_height=height))
    # fill the mask in the result
    label = new_labels[4]
    cv2.fillPoly(result[label], [right_middle_contour.astype(np.int32)], color=(1, 0, 0))
    # top
    # get region contour
    right_bottom_contour, _, _ = (
        get_region_contour(divided_right_indices[1][-1], divided_left_indices[0][0], outer_myo_contour, contour_lv,
                           myo_start_index=myo_end_index4, myo_point_start=myo_point_end4,
                           myo_end_index=epi_apex_index, myo_point_end=epi_apex,
                           img_width=width, img_height=height))
    # fill the mask in the result
    label = new_labels[3]
    cv2.fillPoly(result[label], [right_bottom_contour.astype(np.int32)], color=(1, 0, 0))
    if include_annulus:
        left_annulus_labels = new_labels[6]
        # get location of annulus point and fill a circle around it with radius given by annulus_area_radius
        left_annulus_loc = np.round(base_sep).astype(np.int32)
        result[left_annulus_labels] = cv2.circle(result[left_annulus_labels], left_annulus_loc, annulus_area_radius,
                                                 color=(1, 0, 0), thickness=-1)
        right_annulus_labels = new_labels[7]
        # get location of annulus point and fill a circle around it with radius given by annulus_area_radius
        right_annulus_loc = np.round(base_lat).astype(np.int32)
        result[right_annulus_labels] = cv2.circle(result[right_annulus_labels], right_annulus_loc, annulus_area_radius,
                                                  color=(1, 0, 0), thickness=-1)
    if include_lv_lumen:
        # append the left ventricle lumen mask to the result
        lv_mask = 1*(segmentation == lv_label)
        # add dimension to lv_mask to match number of dimensions in result
        lv_mask = np.expand_dims(lv_mask, axis=0)
        result = np.concatenate((result, lv_mask), axis=0)
    return result