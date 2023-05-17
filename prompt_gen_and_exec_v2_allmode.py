from segment_anything import SamPredictor, sam_model_registry
from PIL import Image, ImageDraw, ImageOps
from shapely.geometry import LineString, MultiLineString, Polygon, Point, GeometryCollection
from skimage.morphology import medial_axis
from scipy.optimize import minimize_scalar
from scipy.ndimage import binary_dilation
from skimage.measure import label
from sklearn.cluster import KMeans

import argparse
import os
import cv2
import json
import imutils
import random
import matplotlib.pyplot as plt
import numpy as np
# Fix randomness in prompt selection
np.random.seed(1)

#This is a helper function that should not be called directly
def _find_closest(centroid, pos_points):
    dist_squared = np.sum((pos_points - centroid)**2, axis=1)
    point_idx = np.argmin(dist_squared)
    return pos_points[point_idx]

def IOU(pm, gt):
    a = np.sum(np.bitwise_and(pm, gt))
    b = np.sum(pm) + np.sum(gt) - a #+ 1e-8 
    if b == 0:
        return -1
    else:
        return a / b

def IOUMulti(y_pred, y):
    score = 0
    numLabels = np.max(y)
    if np.max(y) == 1:
        score = IOU(y_pred, y)
        return score
    else:
        count = 1
        for index in range(1,numLabels+1):
            curr_score = IOU(y_pred[y==index], y[y==index])
            print(index, curr_score)
            if curr_score != -1:
                score += curr_score
                count += 1
        return score / (count - 1) # taking average

####################################################
# input: raw_msk
#   A mask should containing no 'void' class. 
#   Binary mask should have value {0,1} but not {0,255}
# output:
#   A list of region profiles; Each profile takes the form
#   {'loc':[x0,y0,x1,y1], 'cls': cls}
#   'loc' is a list with 4 elements ; 'cls' is object class as integer 
####################################################
def MaskToBoxSimple(mask):
    mask = mask.squeeze()
    #find coordinates of points in the region
    row, col = np.argwhere(mask).T
    # find the four corner coordinates
    y0,x0 = row.min(),col.min()
    y1,x1 = row.max(),col.max()

    return [x0,y0,x1,y1]

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="SAG segmentor for medical images")
    parser.add_argument("--num-prompt", default=1, type=int, help="number of prompts to include, negative number means using box as prompts")
    parser.add_argument("--class-type", default="b", type=str, help="binary or multi class, choose b or m")
    parser.add_argument("--model-path", default="./", type=str, help="the path of the model saved")
    parser.add_argument("--init-path", default="./", type=str, help="the path of the dataset")
    parser.add_argument("--model", default="sam", type=str, help="the model to use as predictor")
    parser.add_argument("--oracle", default=False, type=bool, help="whether eval in the oracle mode, where best prediction is selected based on GT")
    parser.add_argument("--result-image",default="./results",type=str, help="the path to save segmented results")
    parser.add_argument("--result-score",default="./scores",type=str, help="the path to save result metrics")
    args = parser.parse_args()
    
    # Set up model
    sam = sam_model_registry["default"](checkpoint=os.path.join(args.model_path, "sam_vit_h_4b8939.pth"))
    sam.to('cuda')
    predictor = SamPredictor(sam)

    # Set up dataset
    dataset = input("Type of input: ")
    if dataset == 'all':
        # all
        dataset_list = ['busi', 'breast_b', 'breast_d', 'chest', 'gmsc_sp', 'gmsc_gm', 'heart', 'liver', 'petwhole', 'prostate', 'brats_3m', 'xrayhip', \ 
                        'ctliver', 'ctorgan', 'ctcolon', 'cthepaticvessel', 'ctpancreas', 'ctspleen', 'usmuscle', 'usnerve', 'usovariantumor']
    else:
        dataset_list = [dataset]

    for dataset in dataset_list:
        num_class = 1
        if 'gmsc' in dataset:
            input_img_dir = os.path.join(args.init_path, 'sa_gmsc/images') 
            input_seg_dir = os.path.join(args.init_path, 'sa_gmsc/masks')
        elif 'breast' in dataset:
            input_img_dir = "../sa_dbc-2D/imgs"
            if dataset == 'breast_b':
                input_seg_dir = "../sa_dbc-2D/masks_breast"
            else:
                input_seg_dir = "../sa_dbc-2D/masks_dense-tissue"
        else:
            input_img_dir = os.path.join(args.init_path, 'sa_%s/images' % dataset)
            input_seg_dir = os.path.join(args.init_path, 'sa_%s/masks' % dataset)
        
        # Handle dataset with multi-class
        if dataset == 'brats_3m':
            num_class = 3
        if dataset == 'xrayhip':
            num_class = 2
        if dataset == 'ctorgan':
            num_class = 5 

        # target is a variable only used by GMSC
        if dataset == 'gmsc_sp':
            target = 'sp'
        if dataset == 'gmsc_gm':
            target = 'gm'
        print(input_img_dir)
        print(input_seg_dir)

        # Running
        dc_log, names = [], []
        mask_list = os.listdir(input_seg_dir)
        print('# of dataset', len(mask_list))
        
        # VIS: now VIS function is separted into another file. Only provide mask if neede
        vis = False
        # Change to [name1, name2, ...] if only need to run on a few samples
        im_list = None#['CHNCXR_0061_0_mask.png'] 

        for im_idx, im_name in enumerate(mask_list):
            # Skip non-selected images if specified
            print(im_name)
            if im_list is not None:
                if im_name not in im_list:
                    continue

            # GMSC: All masks in the same dir, separated by names
            if 'gmsc' in dataset:
                if target not in im_name:
                    continue

            if 'DS_Store' in im_name:
                continue

            # Read image and mask
            try:
                input_mask = cv2.imread(os.path.join(input_seg_dir, im_name), 0)  
            except:
                print('Cannot read mask', im_name)
                continue
            if np.max(input_mask) == 0:
                print('Empty mask')
                print('*****')
                continue
            
            # In multi-class setting, we assume classes are labeled 0,1,2,3...
            # BraTS has label 1,2,4
            if 'brats' in dataset:
                input_mask[input_mask == 4] = 3
            
            # In binary-class setting, some masks are encoded as 0, 255
            if np.max(input_mask) == 255:
                input_mask = np.uint8(input_mask / input_mask.max())

            # Chest and GMSC: name inconsistentcy
            if 'chest' in dataset:
                im_name = im_name.replace('_mask', '')
            if 'gmsc' in dataset:
                im_name = im_name.replace('mask', 'image').replace(target+'-', '')
            try:
                input_image = Image.open(os.path.join(input_img_dir, im_name)).convert("RGB")
            except:
                print('Cannot read image', im_name)
                continue

            input_array = np.array(input_image)
            input_array = np.uint8(input_array / np.max(input_array) * 255)
            print('Number of labels', np.max(input_mask))
            print('Image maximum', np.max(input_array))
            
            # if we want to do multi-class classification
            # else, we combine all the masks as the same class
            #if args.class_type == 'm':
            if num_class > 1:
                #mask_one_hot = (np.arange(1, input_mask.max()+1) == input_mask[...,None]).astype(int) 
                mask_one_hot = (np.arange(1, num_class+1) == input_mask[...,None]).astype(int) 
            else: 
                mask_one_hot = np.array(input_mask > 0,dtype=int)
            
            if len(mask_one_hot.shape) < 3:
                mask_one_hot = mask_one_hot[:,:,np.newaxis] # height*depth*1, to consistent with multi-class setting
            
            # Start prediction for each class
            predictor.set_image(input_array)
            
            # Mask has to be float
            dc_class_tmp = []
            for cls in range(num_class):
                dc_prompt_tmp = []
                # Cls = 2 means to predict mask with label 3
                # But BraTS use 1,2,4 to label differet classes
                #if cls == 2 and 'brats' in dataset:
                #    cls += 1
                print('Predicting class %s' % cls)
                # segment current class as binary segmentation
                try:
                    mask_cls = np.uint8(mask_one_hot[:,:,cls])
                except:
                    print('Mask do not contain this class, skipped')
                    if num_class == 1:
                        dc_class_tmp.append(np.nan)
                    else:
                        # Fixed with 5 modes for now
                        dc_class_tmp.append([np.nan] * 5)
                    continue

                if np.sum(mask_cls) == 0:
                    print('Empty single cls, skipped')
                    #dc_class_tmp.append(np.nan)
                    if num_class == 1:
                        dc_class_tmp.append(np.nan)
                    else:
                        dc_class_tmp.append([np.nan] * 5)
                    continue
                
                # ------ Generate prompt by our definition -------- #
                preds_mask_full, prompts_full = [], []
                
                # Find all disconnected regions
                label_msk, region_ids = label(mask_cls, connectivity=2, return_num=True)
                print('num of regions found', region_ids)
                ratio_list, regionid_list = [], []
                for region_id in range(1, region_ids+1):
                    #find coordinates of points in the region
                    binary_msk = np.where(label_msk==region_id, 1, 0)

                    # clean some region that is abnormally small
                    r = np.sum(binary_msk) / np.sum(mask_cls)
                    print('curr mask over all mask ratio', r)
                    ratio_list.append(r)
                    regionid_list.append(region_id)

                ratio_list, regionid_list = zip(*sorted(zip(ratio_list, regionid_list)))
                regionid_list = regionid_list[::-1]

                # 5 modes for now
                for mode in range(5):
                    # Mode 0: middle point of LARGEST mask
                    if mode == 0:
                        binary_msk = np.where(label_msk==regionid_list[0], 1, 0)
                        # Calculates the distance to the closest zero pixel for each pixel of the source image.
                        # Ref from RITM: https://github.com/SamsungLabs/ritm_interactive_segmentation/blob/aa3bb52a77129e477599b5edfd041535bc67b259/isegm/data/points_sampler.py
                        # NOTE: numpy and opencv have inverse definition of row and column
                        # NOTE: SAM and opencv have the same definition
                        padded_mask = np.uint8(np.pad(binary_msk, ((1, 1), (1, 1)), 'constant'))
                        dist_img = cv2.distanceTransform(padded_mask, distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)[1:-1, 1:-1]
                        cY, cX = np.where(dist_img==dist_img.max())
                        random_idx = np.random.randint(0, len(cX))
                        cX, cY = int(cX[random_idx]), int(cY[random_idx])

                        prompt = [(cX,cY,1)]
                    # Mode 1: middle point of top-3 LARGEST mask
                    if mode == 1:
                        prompt = []
                        for mask_idx in range(3):
                            if mask_idx < len(regionid_list):
                                binary_msk = np.where(label_msk==regionid_list[mask_idx], 1, 0) 
                                padded_mask = np.uint8(np.pad(binary_msk, ((1, 1), (1, 1)), 'constant'))
                                dist_img = cv2.distanceTransform(padded_mask, distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)[1:-1, 1:-1]
                                cY, cX = np.where(dist_img==dist_img.max())
                                random_idx = np.random.randint(0, len(cX))
                                cX, cY = int(cX[random_idx]), int(cY[random_idx])
                                
                                prompt.append((cX,cY,1))
                    # Mode 2: box of LARGEST mask
                    if mode == 2:
                        binary_msk = np.where(label_msk==regionid_list[0], 1, 0)
                        box = MaskToBoxSimple(binary_msk)
                        prompt = box
                    # Mode 3: box of top-3 LARGEST mask
                    if mode == 3:
                        prompt = []
                        for mask_idx in range(3):
                            if mask_idx < len(regionid_list):
                                binary_msk = np.where(label_msk==regionid_list[mask_idx], 1, 0)
                                box = MaskToBoxSimple(binary_msk)
                                prompt.append(box)
                    # Mode 4: box of ENTIRE mask
                    if mode == 4:
                        box = MaskToBoxSimple(mask_cls)
                        prompt = box

                    # Get output based on prompt type
                    prompt = np.array(prompt)
                    print('mode %s: prompt: %s' % (mode, prompt))
                    if prompt.shape[-1] == 3:
                        pc = prompt[:,:2]
                        pl = prompt[:, -1]
                        preds, _, _ = predictor.predict(point_coords=pc, point_labels=pl)
                    elif prompt.shape[-1] == 4:
                        if len(prompt.shape) == 1:
                            preds, _, _ = predictor.predict(box=prompt)
                        else:
                            preds = None
                            for box in prompt:
                                preds_single, _, _ = predictor.predict(box=box)
                                if preds is None:
                                    preds = preds_single
                                else:
                                    preds += preds_single

                    preds = preds.transpose((1,2,0))
                    if args.oracle:
                        max_slice, max_dc = -1, 0
                        for mask_slice in range(preds.shape[-1]):
                            preds_mask_single = np.array(preds[:,:,mask_slice]>0,dtype=int)
                            dc = IOUMulti(preds_mask_single, mask_cls)
                            if dc > max_dc:
                                max_dc = dc
                                max_slice = mask_slice
                            print(mask_slice, dc)
                        preds_mask_single = np.array(preds[:,:,max_slice]>0,dtype=int)
                    else:
                        preds_mask_single = np.array(preds[:,:,0]>0,dtype=int)

                    dc = IOUMulti(preds_mask_single, mask_cls)
                    dc_prompt_tmp.append(dc)
                    print('IoU:', dc)
                    
                    # Track prediction, only used when vis
                    if vis:
                        preds_mask_full.append(np.expand_dims(preds, 0))
                        prompts_full.append(prompt)

                # assgin final mask for this class to it
                dc_class_tmp.append(dc_prompt_tmp)
            
            dc_log.append(dc_class_tmp)
            names.append(im_name)
            print('****')
            
            # VIS mode only saves mask and prompt information
            if vis:
                # Final shape: N*H*W*3
                # N = number of predictions. 1 if box prompt, otherwise number of prompts
                # H,W = size of mask
                # 3 = number of outputs per prediction. SAM returns 3 outpus per prompt. 
                #     If no oracle mode, select 0
                #     If oracle mode, select maximum slice. 
                #     You can do that later, or use variable "max_slice"
                preds_mask_full = np.concatenate(preds_mask_full)

                # If box:    N*4, N=number of boxes, 4=box coordinate in XYXY format
                # If prompts:N*3, N=number of prmts, 3=cX, cY, pos/neg
                prompts_full = np.array(prompts_full)
                print(preds_mask_full.shape)
                # TODO: replace with desired storage place
                np.save('tmp/%s_pred.npy' % im_name[:-4], preds_mask_full)
                np.save('tmp/%s_prompt.npy' % im_name[:-4], prompts_full)

        if not vis:
            # BRATS labelled class as 1,2,4
            dc_log = np.array(dc_log)
            print(dc_log.shape)
            print(np.nanmean(dc_log, axis=0))
            print(np.nanmean(dc_log))

            version = 'sam_diffmode'
            if args.oracle:
                version += '_oracle'

            json.dump(names, open('scores/v2/%s_binary_names_%s.json' % (version, dataset), 'w+'))
            np.save('scores/v2/%s_binary_score_%s.npy' % (version, dataset), dc_log)


