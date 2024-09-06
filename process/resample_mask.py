import numpy as np
import os
import csv
import warnings
import SimpleITK as sitk
from skimage.transform import resize
from skimage.io import imread
warnings.filterwarnings('ignore')

def readCsv(csvfname):
    # read csv to list of lists
    with open(csvfname, 'r') as csvf:
        reader = csv.reader(csvf)
        csvlines = list(reader)
        print(csvlines[0])
    return csvlines


def imread_mask(name):
    patient_image = sitk.ReadImage(name)
    ori_image = sitk.GetArrayFromImage(patient_image)
    return ori_image

def process_mask(input_path, output_path, mask_type='VAT'):
    """
    Process a mask image and save the resized result.

    Parameters:
    - input_path (str): Path to the input mask image.
    - output_path (str): Path to save the processed mask.
    - mask_type (str): Specify the type of mask ('VAT' or 'SAT').

    Returns:
    None
    """
    mask = imread(input_path)
    if mask_type == 'VAT':
        processed_mask = np.uint8(mask == 1)
    elif mask_type == 'SAT':
        processed_mask = np.uint8(mask == 2)
    else:
        raise ValueError(f"Invalid mask_type. Must be either 'VAT' or 'SAT'.")

    processed_mask = resize(processed_mask, (processed_mask.shape[0], 256, 256), 0, mode='constant', preserve_range=True, cval=0)
    img_save = sitk.GetImageFromArray(processed_mask)
    sitk.WriteImage(img_save, output_path)

if __name__ == "__main__":
    # Input path
    base_path = '../Users/256/'
    base_list = os.listdir(base_path)
    base_list.sort()

    # Output path
    output_path_1 = '../Users/256/'

    for i in range(0,1):
        name = base_path + base_list[i]
        mask = imread_mask(name)
        print(i, base_list[i] + ' is processing...')
        # Process mask
        process_mask(name, output_path_1, mask_type='VAT')



