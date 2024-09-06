"""
Reference：https://blog.csdn.net/JianJuly/article/details/81214408
"""
import SimpleITK as sitk
import os
sep = os.sep

def dcm2nii(file_path):
    # The folder path where the Dicom sequence is located (in our experiment, there are multiple dcm sequences under this folder, mixed together).
    # Obtain all sequence IDs under this file. Each sequence corresponds to an ID. The returned series_IDs is a list.
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(file_path)

    # Check the number of sequences under this folder.
    nb_series = len(series_IDs)
    print(nb_series)

    # Obtain the complete paths of all slices of the sequence corresponding to this ID through the ID. series_IDs[1] represents the ID of the second sequence.
    # If the parameter series_IDs[1] is not added, all slice paths of the first sequence are obtained by default.
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(file_path, series_IDs[0])

    # Create a new ImageSeriesReader object.
    series_reader = sitk.ImageSeriesReader()

    # Read the sequence through the slice paths obtained previously.
    series_reader.SetFileNames(series_file_names)

    # Obtain the 3D image corresponding to this sequence.
    image3D = series_reader.Execute()

    # Convert image to scan.
    # Check the size of this 3D image.
    print(image3D.GetSize())
    sitk.WriteImage(image3D, file_path+sep+'nifti.nii.gz')
    print('save succed')

def research_patient(path_all):
    dir_list = os.listdir(path_all)
    dir_list = [path_all+sep+i for i in dir_list if '.' not in i]

    fail_list = []
    for index,i in enumerate(dir_list):
        print(i,index+1,'/',len(dir_list))

        if not os.path.exists(i+sep+'nifti.nii.gz'):
            try:
                dcm2nii(i)
            except:
                print('failed:', i, 'sum:', len(fail_list))
                fail_list.append(i)
        else:
            print('already turned:',i)

    # Save images that have not been successfully converted.
    with open(path_all+sep+"fail_list.txt","w") as f:
        for i in fail_list:
            f.write(i+'\n')

if __name__ == '__main__':

    # dicom path
    dicom_all = '../data/nas/'
    # Traverse all patient folders under the dicom path.
    print(dicom_all, " is begin....")
    research_patient(dicom_all)














