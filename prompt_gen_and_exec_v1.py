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

import sys
sys.path.append('FocalClick')
#sys.path.append('ritm_interactive_segmentation')
#sys.path.append('CFR-ICL-Interactive-Segmentation')
from isegm.inference.clicker import Click
from isegm.inference import utils as is_utils
from isegm.inference.predictors import get_predictor as is_get_predictor  
from isegm.inference.evaluation import evaluate_sample_onepass as is_evaluate_sample_onepass

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
def MaskToBoxes(mask):
    label_msk, region_ids = label(mask, connectivity=2, return_num=True)
    
    bbox_profiles = []
    for region_id  in range(1, region_ids+1):
        #find coordinates of points in the region
        row,col = np.argwhere(label_msk == region_id).T
        #find class of the region
        cls = mask[row[0],col[0]]
        # find the four corner coordinates
        y0,x0 = row.min(),col.min()
        y1,x1 = row.max(),col.max()

        bbox_profiles.append({'loc':[x0,y0,x1,y1], 'cls':cls})
        
    return bbox_profiles

####################################################
# input: raw_msk
#   A mask should containing no 'void' class. 
#   Binary mask should have value {0,1} but not {0,255}
# input: N
#   The number of points to apply on each object/connected region
# output:
#   A list of region profiles. Each region profile takes the form
#   {'loc':np.array([[x0,y0],[x1,y1],[x_N,y_N]]), 'cls': cls}
#   'loc' is 2D array with shape (N, 2); 'cls' is object class as integer 
####################################################
def Mask2Points(raw_msk, N=1):
    label_msk, region_ids = label(raw_msk, connectivity=2,return_num = True)
    point_profiles = []

    for region_id  in range(1, region_ids+1):
        #find coordinates of points in the region
        pos_points = np.argwhere(label_msk == region_id)
        
        # clean some region that is abnormally small
        r = len(pos_points) / len(raw_msk.flatten())
        if r < 1e-4:
            continue
        print('mask ratio', r)
        #if len(pos_points) < len(raw_msk.flatten())*0.001:
        #    continue
            
        #get the skeleton
        binary_msk = np.where(label_msk == region_id,1,0)
        skeleton_msk = medial_axis(binary_msk).astype(np.uint8)
        skeleton_points = np.argwhere(skeleton_msk>0)

        # Cluster and assign the object skeleton into N sections
        #kmean = KMeans(n_clusters=N,n_init=3, algorithm='lloyd' if N == 1 else 'elkan').fit(skeleton_points)
        kmean = KMeans(n_clusters=N,n_init=3, algorithm='auto').fit(skeleton_points)
        cluster_assigned = np.zeros(len(skeleton_points)) if N == 1 else kmean.predict(skeleton_points)
        centroids = kmean.cluster_centers_
        
        # pick a skeleton point closest to the centroid from each cluster
        selected_points = np.zeros((N,2)) 
        for cluster_id, centroid in zip(range(N),centroids):
            points_in_cluster = skeleton_points[cluster_assigned==cluster_id] 
            selected_points[cluster_id] = _find_closest(centroid,points_in_cluster)
            
        #find class of the region
        cls = raw_msk[pos_points[0,0],pos_points[0,1]]
        
        point_profiles.append({'loc':np.concatenate((selected_points[:,1:],selected_points[:,0:1]),axis=1), 'cls':cls})
        
        #TODO: double check if > 1 regions found
        break
        
    return point_profiles

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
    if args.model == 'sam':
        sam = sam_model_registry["default"](checkpoint=os.path.join(args.model_path, "sam_vit_h_4b8939.pth"))
        sam.to('cuda')
        predictor = SamPredictor(sam)
    # NOTE: manual change sys path when importing library
    elif args.model == 'ritm':
        model = is_utils.load_is_model(os.path.join(args.model_path, "coco_lvis_h32_itermask.pth"), "cuda")
        predictor = is_get_predictor(model, "NoBRS", "cuda")
    elif args.model == 'sc': 
        model = is_utils.load_is_model(os.path.join(args.model_path, "cocolvis_icl_vit_huge.pth"), "cuda", eval_ritm=False)

        zoom_in_params = {
                        'skip_clicks': -1,
                        'target_size': (448, 448)
        }

        predictor_params = {
                        'cascade_step': 4 + 1,
                        'cascade_adaptive': True,
                        'cascade_clicks': 1
        }
        predictor = is_get_predictor(model, "NoBRS", "cuda", prob_thresh=0.49, \
                                     predictor_params=predictor_params, zoom_in_params=zoom_in_params)
    elif args.model == 'fc':
        model = is_utils.load_is_model(os.path.join(args.model_path, "segformerB3_S2_comb.pth"), "cuda")
        predictor = is_get_predictor(model, "NoBRS", "cuda", prob_thresh=0.49)

    print('Dataset you can choose among: chest, gmsc_sp, gmsc_gm, breast_b, breast_f, heart, usbreast, liver, prostate, nodule, brats, all')
    # Set up dataset
    dataset = input("Type of input: ")
    if dataset == 'all':
        dataset_list = ['busi', 'breast_b', 'breast_d', 'chest', 'gmsc_sp', 'gmsc_gm', 'heart', 'liver', 'petwhole', 'prostate', 'brats_3m', 'xrayhip', \ 
                        'ctliver', 'ctorgan', 'ctcolon', 'cthepaticvessel', 'ctpancreas', 'ctspleen', 'usmuscle', 'usnerve', 'usovariantumor']
    else:
        dataset_list = [dataset]

    for dataset in dataset_list:
        print('curr dataset', dataset)
        num_class = 1
        if 'gmsc' in dataset:
            input_img_dir = os.path.join(args.init_path, 'sa_gmsc/images') 
            input_seg_dir = os.path.join(args.init_path, 'sa_gmsc/masks')
        elif 'breast' in dataset:
            input_img_dir = os.path.join(args.init_path, "sa_dbc-2D/imgs")
            if dataset == 'breast_b':
                input_seg_dir = os.path.join(args.init_path, "sa_dbc-2D/masks_breast")
            else:
                input_seg_dir = os.path.join(args.init_path, "sa_dbc-2D/masks_dense-tissue")
        else:
            input_img_dir = os.path.join(args.init_path, 'sa_%s/images' % dataset)
            input_seg_dir = os.path.join(args.init_path, 'sa_%s/masks' % dataset)

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
        
        
        if args.num_prompt<0:
            save_path = os.path.join('results',dataset,'box')
        elif args.oracle:
            save_path = os.path.join('results',dataset,'oracle')
        else:
            save_path = os.path.join('results',dataset,'point')

        # Running
        dc_log, names = [], []
        mask_list = os.listdir(input_seg_dir)
        print('# of dataset', len(mask_list))
        
        # VIS: now VIS function is separted into another file. Only provide mask if needed
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
            if args.model == 'sam':
                predictor.set_image(input_array)
            elif args.model == 'ritm':
                predictor.set_input_image(input_array)
            
            # Mask has to be float
            pre_mask = np.zeros_like(mask_one_hot, dtype=float)
            dc_class_tmp = []
            for cls in range(num_class):
                dc_prompt_tmp = []
                print('Predicting class %s' % cls)
                # segment current class as binary segmentation
                try:
                    mask_cls = np.uint8(mask_one_hot[:,:,cls])
                except:
                    print('Mask do not contain this class, skipped')
                    if num_class == 1:
                        dc_class_tmp.append(np.nan)
                    else:
                        dc_class_tmp.append([np.nan] * args.num_prompt)
                    continue

                if np.sum(mask_cls) == 0:
                    print('Empty single cls, skipped')
                    #dc_class_tmp.append(np.nan)
                    if num_class == 1:
                        dc_class_tmp.append(np.nan)
                    else:
                        dc_class_tmp.append([np.nan] * args.num_prompt)
                    continue
                
                # ------ Generate prompt by SAM's eval protocol -------#
                preds_mask_full, prompts_full,gt_mask_full,input_full = [], [],[],[]

                # Calculates the distance to the closest zero pixel for each pixel of the source image.
                # Ref from RITM: https://github.com/SamsungLabs/ritm_interactive_segmentation/blob/aa3bb52a77129e477599b5edfd041535bc67b259/isegm/data/points_sampler.py
                padded_mask = np.pad(mask_cls, ((1, 1), (1, 1)), 'constant')
                dist_img = cv2.distanceTransform(padded_mask, distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)[1:-1, 1:-1]
                # NOTE: numpy and opencv have inverse definition of row and column
                # NOTE: SAM and opencv have the same definition
                cY, cX = np.where(dist_img==dist_img.max())
                # NOTE: random seems to change DC by +/-1e-4
                # Random sample one point with largest distance
                random_idx = np.random.randint(0, len(cX))
                cX, cY = int(cX[random_idx]), int(cY[random_idx])
                    
                # First point: farthest from the object boundary
                pc = [(cX,cY)]
                pl = [1]

                if args.model == 'sam':
                    preds, _, _ = predictor.predict(point_coords=np.array(pc), point_labels=np.array(pl), return_logits=True)
                elif args.model == 'ritm':
                    # RITM returns mask, mask_prob, iou
                    click_list = [Click(is_positive=True, coords=(cY, cX), indx = 0)]
                    _, preds = is_evaluate_sample_onepass(predictor, click_list)
                    # RITM uses 0.49 as threshold. Substract it to let 0 be the threshold
                    preds = preds - 0.49
                    preds = preds[None,:,:].repeat(3,0)
                elif args.model == 'sc' or args.model == 'fc':
                    # SimpleClick
                    click_list = [Click(is_positive=True, coords=(cY, cX), indx = 0)]
                    _, preds_prob, _ = is_evaluate_sample_onepass(input_array, mask_cls, predictor, click_list, \
                                                                  pred_thr=0.49, iterative=False)
                    preds = preds_prob - 0.49
                    preds = preds[None,:,:].repeat(3,0)
                #elif args.model == 'fc':
                #    click_list = [Click(is_positive=True, coords=(cY, cX), indx = 0)]
                #    _, preds_prob, _ = is_evaluate_sample_onepass(input_array, mask_cls, predictor, click_list, \
                #                                                  pred_thr=0.49, iterative=False)
                #    preds = preds_prob - 0.49

                # if logit < 0, it is more like a background
                preds[preds < 0] = 0 
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
                preds_mask_full.append(np.expand_dims(preds, 0))
                gt_mask_full.append(np.expand_dims(mask_cls, 0))
                input_full.append(input_array)
                prompts_full.append((cX,cY,1))
 
                # Subsequent point: farthest from the boundary of the error region
                for idx_p in range(args.num_prompt - 1):
                    error_mask = np.uint8(np.bitwise_xor(mask_cls, preds_mask_single))
                    padded_mask = np.pad(error_mask, ((1, 1), (1, 1)), 'constant')
                    dist_img = cv2.distanceTransform(padded_mask, distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)[1:-1, 1:-1]
                    cY, cX = np.where(dist_img==dist_img.max())
                    random_idx = np.random.randint(0, len(cX))
                    cX, cY = int(cX[random_idx]), int(cY[random_idx])
                    pc.append((cX, cY))
                    if np.sum(input_mask[cY][cX]) == 0:
                        pl.append(0)
                        prompts_full.append((cX,cY,0))
                    else:
                        pl.append(1)
                        prompts_full.append((cX,cY,1))
                    
                    if args.model == 'sam':
                        preds, _, _ = predictor.predict(point_coords=np.array(pc), point_labels=np.array(pl), return_logits=True)
                    elif args.model == 'ritm':
                        curr_click = Click(is_positive=pl[-1], coords=(cY, cX), indx = idx_p+1)
                        click_list.append(curr_click)
                        _, preds = is_evaluate_sample_onepass(predictor, click_list)
                        preds = preds - 0.49
                        preds = preds[None,:,:].repeat(3,0)
                    elif args.model == 'sc' or args.model == 'fc':
                        curr_click = Click(is_positive=pl[-1], coords=(cY, cX), indx = idx_p+1)
                        click_list.append(curr_click)
                        # SimpleClick
                        _, preds_prob, _ = is_evaluate_sample_onepass(input_array, mask_cls, predictor, click_list, \
                                                                      pred_thr=0.49, iterative=False)
                        preds = preds_prob - 0.49
                        preds = preds[None,:,:].repeat(3,0)

                    # if logit < 0, it is more like a background
                    preds[preds < 0] = 0 
                    preds = preds.transpose((1,2,0))

                    if args.oracle:
                        max_slice, max_dc = -1, 0
                        for mask_slice in range(preds.shape[-1]):
                            preds_mask_single = np.array(preds[:,:,mask_slice]>0,dtype=int)
                            dc = IOUMulti(preds_mask_single, mask_cls)
                            if dc > max_dc:
                                max_dc = dc
                                max_slice = mask_slice
                        preds_mask_single = np.array(preds[:,:,max_slice]>0,dtype=int)
                    else:
                        preds_mask_single = np.array(preds[:,:,0]>0,dtype=int)
                    
                    dc = IOUMulti(preds_mask_single, mask_cls)
                    dc_prompt_tmp.append(dc)

                    preds_mask_full.append(np.expand_dims(preds, 0))
                    gt_mask_full.append(np.expand_dims(mask_cls, 0))
                    input_full.append(input_array)
                print('Final prompts', pc, pl)

                # assgin final mask for this class to it
                print('Predicted DC', dc)
                dc_class_tmp.append(dc_prompt_tmp)
                pre_mask[:,:,cls] = preds[:,:,0]

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
                gt_mask_full = np.concatenate(gt_mask_full)
                input_full = np.concatenate(input_full)
                # If box:    N*4, N=number of boxes, 4=box coordinate in XYXY format
                # If prompts:N*3, N=number of prmts, 3=cX, cY, pos/neg
                prompts_full = np.array(prompts_full)
                print(preds_mask_full.shape)
                # TODO: replace with desired storage place
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                np.save(save_path+'/%s_pred.npy' % im_name[:-4], preds_mask_full)
                np.save(save_path+'/%s_prompt.npy' % im_name[:-4], prompts_full)
                np.save(save_path+'/%s_gt.npy' % im_name[:-4], gt_mask_full)
                np.save(save_path+'/%s_input.npy' % im_name[:-4], input_full)
        
        
        if not vis:
            dc_log = np.array(dc_log)
            print(dc_log.shape)
            print(np.nanmean(dc_log, axis=0))
            print(np.nanmean(dc_log))
                
            version = 'sam_prompt'
            #version = 'sam_oracle'
            #version = 'sam_box'
            if args.model == 'sc':
                version = 'simpleclick'
            if args.model == 'fc':
                version = 'focalclick'
            if args.model == 'ritm':
                version = 'ritm'

            json.dump(names, open('scores/v1_rerun/%s_binary_names_%s.json' % (version, dataset), 'w+'))
            np.save('scores/v1_rerun/%s_binary_score_%s.npy' % (version, dataset), dc_log)


