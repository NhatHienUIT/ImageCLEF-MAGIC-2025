import sys
import glob

import tifffile
import numpy as np

import json


def read_maskfns( folder, imageid2fns_bylabeler, imageids ) :
    for fn in glob.glob( '{}/*.tiff'.format(folder) ) :
        items = fn.split('/')[-1].split('_mask_')
        labeler = items[-1].replace('.tiff','')
        imageid = items[0]
        imageid2fns_bylabeler[labeler] = imageid2fns_bylabeler.get( labeler, {} )
        imageid2fns_bylabeler[labeler][imageid] = imageid2fns_bylabeler[labeler].get( imageid, None )
        imageid2fns_bylabeler[labeler][imageid] = fn
        imageids.add(imageid)


def read_tiffmask( fn ) :
    mask = tifffile.imread(fn)
    return mask.astype(np.uint8)


def get_overlaps( mask1, mask2 ) :
    mask1_flattened = mask1.astype(int).flatten()
    mask2_flattened = mask2.astype(int).flatten()
    tp = np.sum( (( mask1_flattened==mask2_flattened ) & (mask1_flattened==1.0)).astype(int) )
    fn = np.sum( (( mask1_flattened!=mask2_flattened ) & (mask1_flattened==1.0)).astype(int) )
    fp = np.sum( (( mask1_flattened!=mask2_flattened ) & (mask2_flattened==1.0)).astype(int) )
    return tp, fn, fp


def score_masks( imageid2_reffnormask, imageid2_predfnormask, instances ) :
    tp_all = 0
    fn_all = 0
    fp_all = 0

    for imageid in instances :

        mask_gold = get_image( imageid2_reffnormask, imageid )
        mask_sys = get_image( imageid2_predfnormask, imageid )

        tp, fn, fp = get_overlaps( mask_sys, mask_gold )
        tp_all += tp
        fp_all += fp
        fn_all += fn

    jaccard = tp_all/(tp_all+fn_all+fp_all)
    dice = 2 * tp_all / (2*tp_all + fn_all + fp_all )

    results = {'jaccard':jaccard,
               'dice': dice }
    return results


def get_image( imageid2_fnormask, key ) :
    if isinstance( imageid2_fnormask[key], str ) :
        mask = read_tiffmask( imageid2_fnormask[key] )
    else:
        mask = imageid2_fnormask[key]
    return mask


def score_masks_macro( data_jacc, data_dice ) :
    return {
        'jaccard_meanofmax': np.mean( [ np.max(data_jacc[i]) for i in range(len(data_jacc)) ] ),
        'jaccard_meanofmean': np.mean( [ np.mean(data_jacc[i]) for i in range(len(data_jacc)) ] ),
        'dice_meanofmax': np.mean( [ np.max(data_dice[i]) for i in range(len(data_dice)) ] ),
        'dice_meanofmean': np.mean( [ np.mean(data_dice[i]) for i in range(len(data_dice)) ] )
    }


def calculate_perinstance_agreement( imageid2reffns_bylabeler, labelers_gold, imageid2predfns, instances ) :
    data_jaccard = []
    data_dice = []

    for imageid in instances :

        masks_gold = [ get_image( imageid2reffns_bylabeler[labeler],imageid ) for labeler in labelers_gold if ( (labeler in imageid2reffns_bylabeler) and (imageid in imageid2reffns_bylabeler[labeler]) ) ]
        mask_sys = get_image( imageid2predfns, imageid )

        jaccard_overgold = []
        dice_overgold = []
        for mask_gold in masks_gold:
            tp, fn, fp = get_overlaps( mask_sys, mask_gold )
            jaccard = tp/(tp+fn+fp)
            dice = 2 * tp / (2*tp + fn + fp )
            jaccard_overgold.append( jaccard )
            dice_overgold.append( dice )
        
        data_jaccard.append(jaccard_overgold)
        data_dice.append(dice_overgold)

    return data_jaccard, data_dice


def addmajoritvote_mask( imageid2fns_bylabeler, annotators, imageids ) :
    imageid2fns_bylabeler['majorityvote'] = {}
    for imageid in imageids :
        images = [ get_image( imageid2fns_bylabeler[ann],imageid) for ann in annotators if ( (ann in imageid2fns_bylabeler) and (imageid in imageid2fns_bylabeler[ann]) )]
        majority_thresh = np.ceil(len(images)/2)
        mask_sum = images[0]
        for x2 in images[1:] :
            mask_sum = np.add( mask_sum, x2)
        mask = np.where( mask_sum >= majority_thresh, 1, 0 )
        imageid2fns_bylabeler['majorityvote'][imageid] = mask


def main( masks_reference_dir, masks_prediction_dir, sys_suffix) :
    imageid2fn_refs = {}
    imageids_gold = set()
    read_maskfns( masks_reference_dir, imageid2fn_refs, imageids_gold )
    
    imageid2fn_predictions = {}
    imageids_system = set()
    read_maskfns( masks_prediction_dir, imageid2fn_predictions, imageids_system )

    print('Detected {} images with masks for reference.'.format(len(imageids_gold)),file=sys.stderr)
    print('Detected {} images with masks for predictions.'.format(len(imageids_system)),file=sys.stderr)
    
    print('Detected gold labelers: {}'.format(imageid2fn_refs.keys()),file=sys.stderr)
    print('Detected system labelers: {}'.format(imageid2fn_predictions.keys()),file=sys.stderr)

    if len(imageids_system) == 0 :
        return {
            "jaccard_meanofmax": 0.0,
            "jaccard_meanofmean": 0.0,
            "dice_meanofmax": 0.0,
            "dice_meanofmean": 0.0,
            "jaccard": 0.0,
            "dice": 0.0,
            "number_segmentation_instances": 0
        }
          
    #calculate the score 
    data_jacc, data_dice = calculate_perinstance_agreement( imageid2fn_refs, ['ann0','ann1','ann2','ann3'], imageid2fn_predictions[sys_suffix], imageids_gold )
    results = score_masks_macro( data_jacc, data_dice )

    #calculate the score considering 1 gold standard -- the majority vote by pixel
    addmajoritvote_mask( imageid2fn_refs, ['ann0','ann1','ann2','ann3'], imageids_gold )
    results_mv = score_masks( imageid2fn_refs['majorityvote'], imageid2fn_predictions[sys_suffix], imageids_gold )

    results['jaccard'] = results_mv['jaccard']
    results['dice'] = results_mv['dice']
    results['number_segmentation_instances'] = len(imageids_gold)

    return results


if __name__ == "__main__":

    masks_reference_dir = sys.argv[1] if len(sys.argv)>1 else '/app/input/ref/masks_refs'
    masks_prediction_dir = sys.argv[2] if len(sys.argv)>2 else '/app/input/res/masks_preds'
    score_dir = sys.argv[3] if len(sys.argv)>3 else '/app/output/'

    #by default looks for a file {IMGID}_mask_sys.tiff, can change this to something else
    sys_suffix = sys.argv[4] if len(sys.argv)>4 else 'sys'

    results = main( masks_reference_dir, masks_prediction_dir, sys_suffix )

    with open( '{}/scores_segmentation.json'.format(score_dir), 'w' ) as f :
        json.dump( results, f, indent=4 )
