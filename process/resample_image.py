import SimpleITK as sitk
import numpy as np
from scipy import ndimage
import os

def resample(image, spacing, new_spacing):
    spacing = np.array(spacing)
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = ndimage.interpolation.zoom(image, real_resize_factor)

    return image, new_spacing

def my_reshape(image):
    m = np.zeros((384, 512, 464))
    m[26:358, :, :] = image
    return m

def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  # Original voxel block size
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int)

    # Set various properties of the resampler.
    resampler.SetReferenceImage(itkimage)   # Target image to be resampled
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  # Get the resampled image
    return itkimgResampled


if __name__ == "__main__":
    # Input path of the original images.
    root_img_path = '../data/newnas/'
    # Input path of the resampled images.
    root_gtt_path = '../data/newnas/'
    # Set the Z-axis spacing.
    Z_space = 0.625
    for p in os.listdir(root_img_path):
        img_path = os.path.join(root_img_path, p)
        img = sitk.ReadImage(img_path)
        shape = np.shape(sitk.GetArrayFromImage(img))

        # Adjust the size of the resampled image here.
        resize_img = resize_image_itk(img, newSize=(32, 32, shape[0]), resamplemethod=sitk.sitkNearestNeighbor)

        sitk.WriteImage(resize_img, os.path.join(root_gtt_path, p))
        print( os.path.join(root_gtt_path, p))

