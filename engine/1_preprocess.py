from other import judgedir
import h5py
from wama.utils import *
import nibabel as nib
import openpyxl
sep = os.sep
amplitude_amount = np.int32(0)


def Preprocess(imgpath=None, maskpath=None, dst=None, aimsize=[128, 128, -1], sorted_num=-8, pklpath=None):
    '''
    :param src:
    :param dst:
    :param aimsize: [x,y,z], x is the length after cropping, y is the width after cropping.!!!Note!!!: Z is the number of layers expanded in the longitudinal axis (height). By default, it is -1, which means it is the number of layers of the mask/2 rounded up.
    :param sorted_num:
    :param pklpath:
    :return:
    '''
    print('Img2small Beginning')
    # Check paths
    # imgpath = src + sep + 'IMG'   # image
    # maskpath = src + sep + 'MASK'   # mask
    if os.path.exists(imgpath) and os.path.exists(maskpath):
        pass
    else:
        raise ValueError('Path does not exist')
    # Read image files
    if pklpath == None:# No pkl file, traverse according to folder files
        try:
            # Sort file names in ascending numerical order; here x:int(x[:?]);? can be changed as needed. -8 is from the eighth character from the end of the string.
            img = sorted(os.listdir(imgpath), key=lambda x: str(x[:sorted_num]))
            mask = sorted(os.listdir(maskpath), key=lambda x: str(x[:sorted_num]))
        except:
            raise ValueError('Check if there are files or if the file names are of the 12345.nii.gz type')
    else: # With pkl file, read patient ids in the file
        img = [] # Mimic no Pkl
        pkl = open(pklpath, 'rb')
        alldata_label = pickle.load(pkl) # Read pkl
        for pth_id in alldata_label.keys():
            if alldata_label[pth_id]['Use'] == 1:
                img.append(str(pth_id)+'.nii.gz')
        mask = img
        img = sorted(os.listdir(imgpath), key=lambda x: str(x[:sorted_num]))
        mask = img

    # Loop through each patient
    flag = 0
    for imgname, maskname in zip(img, mask):
        flag = flag+1
        print('Progress:' + str(flag) + ' / ' + str(len(img)) + ' || Current combination: ' + str(imgname) + ' | ' + str(maskname))
        imageFilepath = os.path.join(imgpath, imgname) # Original image path
        maskFilepath = os.path.join(maskpath, maskname) # Mask path
        if not os.path.exists(imageFilepath):
            continue
        if os.path.exists(os.path.join(dst, 'preprocess_imagedata', 'IMG', imgname)):
            continue
        # Find the boxC
        subject = wama()
        task_name = 'CT'
        subject.appendImageFromNifti(task_name, imageFilepath)
        subject.appendSementicMaskFromNifti(task_name, maskFilepath)
        img_box = subject.getBbox(task_name)
        # Center point location
        centerX = int((img_box[1] - img_box[0]) / 2) + img_box[0]
        centerY = int((img_box[3] - img_box[2]) / 2) + img_box[2]

        if aimsize[2] == -1:
            aimsize[2] = math.ceil((img_box[5]-img_box[4])/2)
        if aimsize[0] == aimsize[1] and aimsize[0] % 2 == 0:
            minx = int(centerX - aimsize[0] / 2)
            maxx = int(centerX + aimsize[0] / 2)
            miny = int(centerY - aimsize[1] / 2)
            maxy = int(centerY + aimsize[1] / 2)
            minz = int(img_box[4] - aimsize[2])
            maxz = int(img_box[5] + aimsize[2])
            # Handle out-of-bounds cases
            if minx < 0:
                minx, maxx = 0, aimsize[0]
            if maxx > subject.scan[task_name].shape[0]:
                minx, maxx = subject.scan[task_name].shape[0]-aimsize[0], subject.scan[task_name].shape[0]
            if miny < 0:
                miny, maxy = 0, aimsize[1]
            if maxy > subject.scan[task_name].shape[1]:
                miny, maxy = subject.scan[task_name].shape[1]-aimsize[1], subject.scan[task_name].shape[1]
            minz, maxz = np.max([minz, 0]), np.min([maxz, subject.scan[task_name].shape[2]])

        # Save bbox nii image
        imgsave = subject.scan[task_name][minx:maxx, miny:maxy, minz:maxz]
        masksave = subject.sementic_mask[task_name][minx:maxx, miny:maxy, minz:maxz]
        # Check
        if imgsave.shape[0]!= aimsize[0] or imgsave.shape[1]!= aimsize[1]:
            print(imgname, 'Please check size!')
        # Transpose
        imgsave_nonorma = np.transpose(imgsave, (2,1,0))
        masksave = np.transpose(masksave, (2,1,0))

        # Normalization
        pos = np.unravel_index(np.argmax(imgsave_nonorma), imgsave_nonorma.shape)  # Return coordinates of maximum value
        max_data = imgsave_nonorma[pos[0]][pos[1]][pos[2]]  # Get maximum value
        pos = np.unravel_index(np.argmin(imgsave_nonorma), imgsave_nonorma.shape)  # Return coordinates of minimum value
        min_data = imgsave_nonorma[pos[0]][pos[1]][pos[2]]  # Get minimum value
        imgsave = ((imgsave_nonorma - min_data) / (max_data - min_data)).astype('float32')  # Overall normalization

        # Save parameters
        final_img = sitk.GetImageFromArray(imgsave)
        final_img.SetSpacing(subject.spacing[task_name])  # voxelsize
        final_img.SetOrigin(subject.origin[task_name])  # world coordinates of origin
        final_img.SetDirection(subject.transfmat[task_name])  # 3D rotation matrix
        final_mask = sitk.GetImageFromArray(masksave)
        final_mask.SetSpacing(subject.spacing[task_name])  # voxelsize
        final_mask.SetOrigin(subject.origin[task_name])  # world coordinates of origin
        final_mask.SetDirection(subject.transfmat[task_name])  # 3D rotation matrix
        judgedir(os.path.join(dst, 'preprocess_imagedata', 'IMG'))
        judgedir(os.path.join(dst, 'preprocess_imagedata', 'MASK'))
        sitk.WriteImage(final_img, os.path.join(dst, 'preprocess_imagedata', 'IMG', imgname))
        sitk.WriteImage(final_mask, os.path.join(dst, 'preprocess_imagedata', 'MASK', maskname))

        # Check the number of layers of the mask and img
        if imgsave.shape[0]!= masksave.shape[0]:
            raise ValueError('Check if the number of layers of the sliced image {} is consistent'.format(imgname))
        else:
            pass

        # Save slices (use this part when generating h5 files)
        z, x, y = imgsave.shape
        for zz in range(z):
            # Slice along the z-axis (transverse position), so it is z.
            slice_img = imgsave[zz, :, :] # Number of z-axis layers of img
            slice_mask = masksave[zz, :, :] # Number of z-axis layers of mask
            try:
                pth_name = imgname.split('.')[0]
                judgedir(os.path.join(dst, 'preprocess_h5data', pth_name))
                save_name = os.path.join(dst, 'preprocess_h5data', pth_name, str(pth_name) + '_' + str(zz+1) + '.h5')
            except:
                raise ValueError('Please ensure that the string before the first delimiter of the file name is a number')
            # Save img and mask as np format to the.h5 file
            if os.path.exists(save_name) == 0:
                f = h5py.File(save_name, 'a')
                f['image'] = slice_img.astype('float32')
                f['roi'] = slice_mask.astype('float32')
                # Different factors of DSI
                f['label1'] = alldata_label[pth_name]['Label1']
                f['label2'] = alldata_label[pth_name]['Label2']
                f['label3'] = alldata_label[pth_name]['Label3']
                f['label12'] = alldata_label[pth_name]['Label12']
                f['label13'] = alldata_label[pth_name]['Label13']
                f['label14'] = alldata_label[pth_name]['Label14']
                f['label15'] = alldata_label[pth_name]['Label15']
                f['label16'] = alldata_label[pth_name]['Label16']
                f['labelall'] = alldata_label[pth_name]['Labelall']
                f.close()
            else:
                os.remove(save_name)
                f = h5py.File(save_name, 'a')
                f['image'] = slice_img.astype('float32')
                f['roi'] = slice_mask.astype('float32')
                # Different factors of DSI
                f['label1'] = alldata_label[pth_name]['Label1']
                f['label2'] = alldata_label[pth_name]['Label2']
                f['label3'] = alldata_label[pth_name]['Label3']
                f['label12'] = alldata_label[pth_name]['Label12']
                f['label13'] = alldata_label[pth_name]['Label13']
                f['label14'] = alldata_label[pth_name]['Label14']
                f['label15'] = alldata_label[pth_name]['Label15']
                f['label16'] = alldata_label[pth_name]['Label16']
                f['labelall'] = alldata_label[pth_name]['Labelall']
                f.close()

def h5_generate(imgpath=None, maskpath=None, dst=None, sorted_num=-8, pklpath=None):
    print('H5 Beginning')
    # Check paths
    if os.path.exists(imgpath) and os.path.exists(maskpath):
        pass
    else:
        raise ValueError('Path does not exist')
    # Read image files

    # Sort file names in ascending numerical order; here x:int(x[:?]);? can be changed as needed. -8 is from the eighth character from the end of the string.
    img = sorted(os.listdir(imgpath), key=lambda x: str(x[:sorted_num]))
    mask = sorted(os.listdir(maskpath), key=lambda x: str(x[:sorted_num]))

    pkl = open(pklpath, 'rb')
    alldata_label = pickle.load(pkl) # Read pkl

    # Loop through each patient
    flag = 0
    for imgname, maskname in zip(img, mask):
        flag = flag+1
        print('Progress:' + str(flag) + ' / ' + str(len(img)) + ' || Current combination: ' + str(imgname) + ' | ' + str(maskname))
        imageFilepath = os.path.join(imgpath, imgname) # Original image path
        maskFilepath = os.path.join(maskpath, maskname) # Mask path
        if os.path.exists(os.path.join(dst, 'preprocess_imagedata', 'IMG', imgname)):
            continue
        if not os.path.exists(imageFilepath) or not os.path.exists(maskFilepath):
            print("{} file does not exist!!!".format(imageFilepath))

        # Read nii.gz files
        imgsave = nib.load(imageFilepath).get_fdata() # Read nii.gz file
        masksave = nib.load(maskFilepath).get_fdata() # Read nii.gz file

        # Save slices (use this part when generating h5 files)
        x, y, z= imgsave.shape
        for zz in range(z):
            # Slice along the z-axis (transverse position), so it is z.
            slice_img = imgsave[ :, :, zz]  # Number of z-axis layers of img
            slice_mask = masksave[ :, :, zz]  # Number of z-axis layers of mask
            try:
                pth_name = imgname.split('.')[0]
                judgedir(os.path.join(dst, 'preprocess_h5data', pth_name))
                save_name = os.path.join(dst, 'preprocess_h5data', pth_name, str(pth_name) + '_' + str(zz + 1) + '.h5')
            except:
                raise ValueError('Please ensure that the string before the first delimiter of the file name is a number')
            # Save img and mask as np format to the.h5 file
            if os.path.exists(save_name) == 0:
                f = h5py.File(save_name, 'a')
                f['image'] = slice_img.astype('float32')
                f['roi'] = slice_mask.astype('float32')
                # Different factors of DSI
                f['label1'] = alldata_label[pth_name]['Label1']
                f['label2'] = alldata_label[pth_name]['Label2']
                f['label3'] = alldata_label[pth_name]['Label3']
                f['label12'] = alldata_label[pth_name]['Label12']
                f['label13'] = alldata_label[pth_name]['Label13']
                f['label14'] = alldata_label[pth_name]['Label14']
                f['label15'] = alldata_label[pth_name]['Label15']
                f['label16'] = alldata_label[pth_name]['Label16']
                f['labelall'] = alldata_label[pth_name]['Labelall']
                f.close()
            else:
                os.remove(save_name)
                f = h5py.File(save_name, 'a')
                f['image'] = slice_img.astype('float32')
                f['roi'] = slice_mask.astype('float32')
                # Different factors of DSI
                f['label1'] = alldata_label[pth_name]['Label1']
                f['label2'] = alldata_label[pth_name]['Label2']
                f['label3'] = alldata_label[pth_name]['Label3']
                f['label12'] = alldata_label[pth_name]['Label12']
                f['label13'] = alldata_label[pth_name]['Label13']
                f['label14'] = alldata_label[pth_name]['Label14']
                f['label15'] = alldata_label[pth_name]['Label15']
                f['label16'] = alldata_label[pth_name]['Label16']
                f['labelall'] = alldata_label[pth_name]['Labelall']
                f.close()


def Excel_to_pkl(src,dst=None):
    '''
    Convert the divided training and test set excel files into pkl format. Reading pkl is much faster than reading excel.
    :param src: Path of excel.xlsx file
    :param dst: Folder
    '''
    # If no save path is set, the pkl file is saved by default in the folder of the excel.
    if dst == None:
        dst = src.strip('datatopkl.xlsx')
    print('Excel_to_pkl Beginning!')

    det_name = src.split(sep)[-1].split('.')[0] # Get excel file name
    dst_name = det_name+'.pkl'

    data = openpyxl.load_workbook(src)
    ws = data.active

    sheet1 = data.worksheets[0] # sheet1
    prj = sheet1.rows
    prjList = list(prj)


    ID, Label, Type, Use, Seg = ([] for i in range(5))
    for idx in range(1, len(prjList)):
        list_all = list(prjList[idx])
        ID.append(list_all[0].value)
        Label.append(list_all[1].value)
        Type.append(list_all[2].value)
        Use.append(list_all[3].value)
        Seg.append(list_all[4].value)

    datas = dict([])  # Create an empty dictionary
    for x in range(len(ID)):
        id = str(ID[x])
        label = int(Label[x])
        type = str(Type[x])
        use = int(Use[x])
        seg = int(Seg[x])
        xx = {id: {'Label': label, 'Type': type, 'Use': use, 'Seg': seg}}
        datas.update(xx)  # Update the dictionary
    with open(os.path.join(dst, dst_name), 'wb') as fo:  # Write data to pkl file
        pickle.dump(datas, fo)
    return os.path.join(dst, dst_name)