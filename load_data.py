import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os 
import glob
import pydicom
import scipy.ndimage



# preprocessing 
# Modified from https://github.com/bdrad/Tom_NLST_utils/blob/main/NLST_image_Processing_utils_research_drive.ipynb

def resize(img, img_dimension):
    '''
    Resize 3D image to the desired dimensions. 
    :Parameters
    img: 3D image to be resized 
    img_dimension: a tuple of desired dimension, [desired num slice, desired height, desired weight
    '''
    num_slice, height, width = img.shape
    desired_thickness, desired_height, desired_width = img_dimension

    slice_thickness_factor = desired_thickness / num_slice  
    height_factor = desired_height / height
    width_factor = desired_width / width

    new_ct_img = scipy.ndimage.interpolation.zoom(img, [slice_thickness_factor, height_factor, width_factor])

    return new_ct_img


def load_dicom(folder_location, num_slice, img_dimension, windowing=True, window_center=-600, window_width=1500):
    '''
    :function 
    Load dicom images in one folder, give them correct ordering, convert to Hounsfield units,
    downsample-making sure images are of the same size, and windowing (consistent across all images).

    :returns
    A slice of tuple (preprocessed dicom file, dicom file's attribute SliceLocation)

    :params 
    img_dimension: if equals NONE, no resizing is done, 
        else is a tuple (desired height x axis = 0, desired width y axis = 1), 
        which is the the dimensions to output each 2D slice in a CT volume
    windowing: if TRUE, use params window_center and window_width to implement windowing 
    window_center: window center parameter for windowing 
    window length: window level parameter for windowing 
    '''
    # retrieve names of all dcm files
    dicom_names = glob.glob(os.path.join(folder_location,'*.dcm'))
    slices = []
    for name in dicom_names: 
        # load dcm files
        dcm = pydicom.dcmread(name)
        # convert to HU units
        img = float(dcm.RescaleSlope) * dcm.pixel_array + float(dcm.RescaleIntercept)


        # windowing 
        if windowing:
            img_min = window_center - window_width // 2 # minimum HU level
            img_max = window_center + window_width // 2 # maximum HU level
            img[img<img_min] = img_min #set img_min for all HU levels less than minimum HU level
            img[img>img_max] = img_max #set img_max for all HU levels higher than maximum HU level

        slices.append((img, dcm.SliceLocation))

    # order the slices
    slices = sorted(slices, key = lambda x: x[1], reverse=True)

    all_slice_img = np.array([s[0] for s in slices])

    # resize 
    if img_dimension is not None:
        all_slice_img = resize(all_slice_img, (num_slice, img_dimension, img_dimension))

    # all_slice_img = resize_image(all_slice_img, spacings)
    return all_slice_img


def main():
    print("Hello World!")

if __name__ == "__main__":
    main()