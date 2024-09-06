import os
import re
import h5py
import numpy as np
import SimpleITK as sitk
from other import judgedir
from radiomics import featureextractor as FEE
import pandas as pd
sep = os.sep

# image和mask文件对齐
def image_mask_align(imgpath, maskpath, output_mask_file):
    img_path = sorted(os.listdir(imgpath))
    mask_path = sorted(os.listdir(maskpath))
    for imgname, maskname in zip(img_path, mask_path):
        if(imgname!= maskname):
            print("image and mask are not aligned!")
            return
        # Read image and mask
        image = sitk.ReadImage(os.path.join(imgpath, imgname))
        mask = sitk.ReadImage(os.path.join(maskpath, maskname))

        # Get the physical size and center point of the image
        image_spacing = image.GetSpacing()
        image_origin = image.GetOrigin()
        image_direction = image.GetDirection()

        # Set the physical size, center point and direction of the mask to be the same as the image
        mask.SetSpacing(image_spacing)
        mask.SetOrigin(image_origin)
        mask.SetDirection(image_direction)

        # Align the mask and the image
        mask_aligned = sitk.Resample(mask, image)

        # Save the aligned mask
        print("file is saved here: {}".format(output_mask_file + imgname))
        sitk.WriteImage(mask_aligned, output_mask_file + imgname)

def RadioFeature(imgpath, maskpath, savefeaturepath, yamlpath):
    '''
    Radiomics feature extraction
    :param Imgpath:
    :param savefeaturepath:
    :params_labels.yaml path:
    :return:
    '''
    judgedir(savefeaturepath, RemoveFlag=True)  # Check if the save folder exists and create it if not
    flag = 0
    img = sorted(os.listdir(imgpath))
    mask = sorted(os.listdir(maskpath))

    df = pd.DataFrame() # Save features as an excel file

    for imgname, maskname in zip(img, mask):

        try:
            flag = flag + 1
            name = re.split(r'[_]+', imgname)[0]
            name_check = re.split(r'[_]+', maskname)[0]
            if name == name_check:
                if os.path.exists(os.path.join(savefeaturepath, '%s.h5' % name)):
                    pass
                else:
                    imageFilepath = os.path.join(imgpath, imgname)
                    maskFilepath = os.path.join(maskpath, maskname)
                    print('Progress:' + str(flag) + ' / ' + str(len(img)) + '||   Current combination: ' + str(imgname) + ' | ' + str(maskname))
                    extractor = FEE.RadiomicsFeatureExtractor(yamlpath)  # todo Initialize feature extractor
                    extractor.loadParams(yamlpath)  # todo Load yaml file
                    result = extractor.execute(imageFilepath, maskFilepath)
                    feature = [value for key, value in result.items() if not 'diagnostics' in key]


                    value_array = np.array(feature)   # Convert list features to ndarray
                    value_array = value_array.reshape(1, -1)
                    with h5py.File(os.path.join(savefeaturepath, '%s.h5' % name), mode='w') as f:
                        f.create_dataset('f_values', data=value_array)
                    f.close()

                    # # Save as excel
                    result.update({'name': name}) # Save patient name in a certain column of excel
                    df_new = pd.DataFrame.from_dict(result.values()).T
                    df_new.columns = result.keys()
                    df = pd.concat([df, df_new])
            else:
                raise ValueError("ctname: ", imgname, "roiname:", maskname)
        except:
            print(imgname,'has a problem')
    df.to_excel(os.path.join(savefeaturepath, 'feature.xlsx'))