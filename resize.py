import sys
import glob
import os

import tifffile
import numpy as np
from scipy.ndimage import zoom  # For resizing masks
import json


def read_maskfns(folder, imageid2fns_bylabeler, imageids):
    print(f"Scanning directory: {folder}")
    files_found = glob.glob(f'{folder}/*.tiff')
    print(f"Found {len(files_found)} .tiff files in {folder}")
    
    for fn in files_found:
        print(f"Processing file: {fn}")
        base_filename = os.path.basename(fn)
        items = base_filename.split('_mask_')
        
        if len(items) != 2:
            print(f"WARNING: File {fn} doesn't match expected naming pattern (IMAGEID_mask_LABELER.tiff)")
            continue
            
        labeler = items[-1].replace('.tiff','')
        imageid = items[0]
        
        print(f"  Extracted - ImageID: {imageid}, Labeler: {labeler}")
        
        if labeler not in imageid2fns_bylabeler:
            imageid2fns_bylabeler[labeler] = {}
            
        imageid2fns_bylabeler[labeler][imageid] = fn
        imageids.add(imageid)
    
    print(f"Detected labelers: {list(imageid2fns_bylabeler.keys())}")


def read_tiffmask(fn):
    try:
        mask = tifffile.imread(fn)
        # Check for RGB/multi-channel mask and convert to binary single channel
        if len(mask.shape) > 2:
            print(f"Converting multi-channel mask to binary: {fn}, shape: {mask.shape}")
            # If it's RGB, take first channel or average, then threshold
            if mask.shape[2] == 3:  # RGB format
                # Take first channel as a simple approach
                mask = mask[:, :, 0]
            else:
                # Average across channels
                mask = np.mean(mask, axis=2)
        
        # Ensure binary (0 or 1)
        mask = (mask > 0).astype(np.uint8)
        print(f"Final mask shape: {mask.shape}, values: min={mask.min()}, max={mask.max()}")
        return mask
        
    except Exception as e:
        print(f"Error reading mask file {fn}: {e}")
        # Return a small empty mask as fallback
        return np.zeros((10, 10), dtype=np.uint8)


def resize_mask(mask, target_shape):
    """Resize a mask to match the target shape"""
    if mask.shape == target_shape:
        return mask
        
    print(f"Resizing mask from {mask.shape} to {target_shape}")
    
    # Calculate zoom factors for each dimension
    factors = [float(target) / float(source) for target, source in zip(target_shape, mask.shape)]
    
    # Resize using scipy's zoom function
    resized = zoom(mask, factors, order=0)  # order=0 ensures binary values stay binary
    
    # Ensure the result is still binary (0s and 1s)
    resized = (resized > 0.5).astype(np.uint8)
    
    return resized


def get_overlaps(mask1, mask2):
    """Calculate TP, FN, FP between two binary masks, resizing if needed"""
    # Ensure both are 2D single-channel binary masks
    if len(mask1.shape) > 2:
        print(f"WARNING: mask1 has shape {mask1.shape}, converting to 2D")
        if mask1.shape[2] == 3:  # RGB format
            mask1 = mask1[:, :, 0]  # Take first channel
        else:
            mask1 = np.mean(mask1, axis=2)
        mask1 = (mask1 > 0).astype(np.uint8)
        
    if len(mask2.shape) > 2:
        print(f"WARNING: mask2 has shape {mask2.shape}, converting to 2D")
        if mask2.shape[2] == 3:  # RGB format
            mask2 = mask2[:, :, 0]  # Take first channel
        else:
            mask2 = np.mean(mask2, axis=2)
        mask2 = (mask2 > 0).astype(np.uint8)
    
    # Check if masks have the same dimensions
    if mask1.shape != mask2.shape:
        print(f"WARNING: Masks have different shapes: {mask1.shape} vs {mask2.shape}")
        # Choose the larger shape as target
        if mask1.size > mask2.size:
            mask2 = resize_mask(mask2, mask1.shape)
        else:
            mask1 = resize_mask(mask1, mask2.shape)
    
    # Debug info
    print(f"Final mask comparison shapes: {mask1.shape} vs {mask2.shape}")
    
    mask1_flattened = mask1.astype(int).flatten()
    mask2_flattened = mask2.astype(int).flatten()
    
    tp = np.sum(((mask1_flattened == mask2_flattened) & (mask1_flattened == 1)).astype(int))
    fn = np.sum(((mask1_flattened != mask2_flattened) & (mask1_flattened == 1)).astype(int))
    fp = np.sum(((mask1_flattened != mask2_flattened) & (mask2_flattened == 1)).astype(int))
    
    print(f"Overlap stats: TP={tp}, FN={fn}, FP={fp}")
    return tp, fn, fp


def score_masks(imageid2_reffnormask, imageid2_predfnormask, instances):
    tp_all = 0
    fn_all = 0
    fp_all = 0

    for imageid in instances:
        try:
            print(f"Processing image: {imageid}")
            mask_gold = get_image(imageid2_reffnormask, imageid)
            mask_sys = get_image(imageid2_predfnormask, imageid)

            tp, fn, fp = get_overlaps(mask_sys, mask_gold)
            tp_all += tp
            fp_all += fp
            fn_all += fn
        except Exception as e:
            print(f"Error processing image {imageid}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    jaccard = tp_all / (tp_all + fn_all + fp_all) if (tp_all + fn_all + fp_all) > 0 else 0
    dice = 2 * tp_all / (2 * tp_all + fn_all + fp_all) if (2 * tp_all + fn_all + fp_all) > 0 else 0

    print(f"Final scores: Jaccard={jaccard}, Dice={dice}")
    results = {'jaccard': jaccard,
               'dice': dice}
    return results


def get_image(imageid2_fnormask, key):
    if key not in imageid2_fnormask:
        print(f"WARNING: Image ID '{key}' not found in mask dictionary")
        return np.zeros((10, 10), dtype=np.uint8)  # Return empty mask as fallback
        
    if isinstance(imageid2_fnormask[key], str):
        mask = read_tiffmask(imageid2_fnormask[key])
    else:
        mask = imageid2_fnormask[key]
    return mask


def score_masks_macro(data_jacc, data_dice):
    if not data_jacc or len(data_jacc) == 0:
        print("WARNING: No data for Jaccard calculation")
        return {
            'jaccard_meanofmax': 0.0,
            'jaccard_meanofmean': 0.0,
            'dice_meanofmax': 0.0,
            'dice_meanofmean': 0.0
        }
        
    print(f"Calculating macro scores from {len(data_jacc)} instances")
    
    # Safe calculation with error handling
    try:
        jaccard_max_values = [np.max(data_jacc[i]) for i in range(len(data_jacc)) if len(data_jacc[i]) > 0]
        jaccard_mean_values = [np.mean(data_jacc[i]) for i in range(len(data_jacc)) if len(data_jacc[i]) > 0]
        dice_max_values = [np.max(data_dice[i]) for i in range(len(data_dice)) if len(data_dice[i]) > 0]
        dice_mean_values = [np.mean(data_dice[i]) for i in range(len(data_dice)) if len(data_dice[i]) > 0]
        
        jaccard_meanofmax = np.mean(jaccard_max_values) if jaccard_max_values else 0.0
        jaccard_meanofmean = np.mean(jaccard_mean_values) if jaccard_mean_values else 0.0
        dice_meanofmax = np.mean(dice_max_values) if dice_max_values else 0.0
        dice_meanofmean = np.mean(dice_mean_values) if dice_mean_values else 0.0
        
    except Exception as e:
        print(f"Error in score_masks_macro: {e}")
        jaccard_meanofmax = jaccard_meanofmean = dice_meanofmax = dice_meanofmean = 0.0
        
    return {
        'jaccard_meanofmax': jaccard_meanofmax,
        'jaccard_meanofmean': jaccard_meanofmean,
        'dice_meanofmax': dice_meanofmax,
        'dice_meanofmean': dice_meanofmean
    }


def calculate_perinstance_agreement(imageid2reffns_bylabeler, labelers_gold, imageid2predfns, instances):
    data_jaccard = []
    data_dice = []

    for imageid in instances:
        try:
            print(f"\nCalculating agreement for image: {imageid}")
            
            # Get gold masks
            masks_gold = []
            for labeler in labelers_gold:
                if labeler in imageid2reffns_bylabeler and imageid in imageid2reffns_bylabeler[labeler]:
                    print(f"  Loading gold mask from labeler: {labeler}")
                    mask = get_image(imageid2reffns_bylabeler[labeler], imageid)
                    masks_gold.append(mask)
            
            if not masks_gold:
                print(f"  WARNING: No gold masks found for image {imageid}")
                continue
                
            # Get system mask
            print(f"  Loading system prediction mask")
            mask_sys = get_image(imageid2predfns, imageid)
            
            jaccard_overgold = []
            dice_overgold = []
            for i, mask_gold in enumerate(masks_gold):
                print(f"  Comparing with gold mask #{i+1}")
                tp, fn, fp = get_overlaps(mask_sys, mask_gold)
                jaccard = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0
                dice = 2 * tp / (2 * tp + fn + fp) if (2 * tp + fn + fp) > 0 else 0
                print(f"  Scores: Jaccard={jaccard}, Dice={dice}")
                jaccard_overgold.append(jaccard)
                dice_overgold.append(dice)
            
            data_jaccard.append(jaccard_overgold)
            data_dice.append(dice_overgold)
            
        except Exception as e:
            print(f"Error processing agreement for image {imageid}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    return data_jaccard, data_dice


def addmajoritvote_mask(imageid2fns_bylabeler, annotators, imageids):
    imageid2fns_bylabeler['majorityvote'] = {}
    for imageid in imageids:
        try:
            print(f"\nCreating majority vote mask for image: {imageid}")
            
            # Get masks from annotators
            images = []
            for ann in annotators:
                if ann in imageid2fns_bylabeler and imageid in imageid2fns_bylabeler[ann]:
                    print(f"  Loading mask from annotator: {ann}")
                    mask = get_image(imageid2fns_bylabeler[ann], imageid)
                    
                    # Ensure 2D binary mask
                    if len(mask.shape) > 2:
                        print(f"  Converting {ann}'s mask from shape {mask.shape} to 2D")
                        if mask.shape[2] == 3:  # RGB format
                            mask = mask[:, :, 0]  # Take first channel
                        else:
                            mask = np.mean(mask, axis=2)
                        mask = (mask > 0).astype(np.uint8)
                        
                    images.append(mask)
            
            if not images:
                print(f"  WARNING: No annotator masks found for image {imageid}, skipping majority vote")
                continue
            
            # Make sure all images have the same shape by resizing to the first one's shape
            reference_shape = images[0].shape
            for i in range(1, len(images)):
                if images[i].shape != reference_shape:
                    print(f"  Resizing annotator mask #{i+1} from {images[i].shape} to {reference_shape}")
                    images[i] = resize_mask(images[i], reference_shape)
                
            majority_thresh = np.ceil(len(images) / 2)
            print(f"  Majority threshold: {majority_thresh} out of {len(images)} annotators")
            
            mask_sum = images[0].copy()
            for i, x2 in enumerate(images[1:], 1):
                print(f"  Adding annotator mask #{i+1} to sum")
                mask_sum = np.add(mask_sum, x2)
                
            mask = np.where(mask_sum >= majority_thresh, 1, 0)
            print(f"  Created majority vote mask with shape {mask.shape}")
            
            imageid2fns_bylabeler['majorityvote'][imageid] = mask
            
        except Exception as e:
            print(f"Error creating majority vote for image {imageid}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue


def main(masks_reference_dir, masks_prediction_dir, sys_suffix):
    imageid2fn_refs = {}
    imageids_gold = set()
    read_maskfns(masks_reference_dir, imageid2fn_refs, imageids_gold)
    
    imageid2fn_predictions = {}
    imageids_system = set()
    read_maskfns(masks_prediction_dir, imageid2fn_predictions, imageids_system)

    print('Detected {} images with masks for reference.'.format(len(imageids_gold)), file=sys.stderr)
    print('Detected {} images with masks for predictions.'.format(len(imageids_system)), file=sys.stderr)
    
    print('Detected gold labelers: {}'.format(list(imageid2fn_refs.keys())), file=sys.stderr)
    print('Detected system labelers: {}'.format(list(imageid2fn_predictions.keys())), file=sys.stderr)

    if len(imageids_system) == 0:
        print("WARNING: No prediction images found", file=sys.stderr)
        return {
            "jaccard_meanofmax": 0.0,
            "jaccard_meanofmean": 0.0,
            "dice_meanofmax": 0.0,
            "dice_meanofmean": 0.0,
            "jaccard": 0.0,
            "dice": 0.0,
            "number_segmentation_instances": 0
        }
    
    # Check if the specified system labeler exists
    if sys_suffix not in imageid2fn_predictions:
        print(f"ERROR: System labeler '{sys_suffix}' not found in predictions", file=sys.stderr)
        print(f"Available labelers: {list(imageid2fn_predictions.keys())}", file=sys.stderr)
        
        # If we have at least one labeler, use the first one as fallback
        if len(imageid2fn_predictions) > 0:
            first_labeler = list(imageid2fn_predictions.keys())[0]
            print(f"Falling back to first available labeler: '{first_labeler}'", file=sys.stderr)
            sys_suffix = first_labeler
        else:
            print("No labelers found in predictions, cannot continue", file=sys.stderr)
            return {
                "jaccard_meanofmax": 0.0,
                "jaccard_meanofmean": 0.0,
                "dice_meanofmax": 0.0,
                "dice_meanofmean": 0.0,
                "jaccard": 0.0,
                "dice": 0.0,
                "number_segmentation_instances": 0,
                "error": "No prediction labelers found"
            }
    
    # Calculate the score 
    data_jacc, data_dice = calculate_perinstance_agreement(
        imageid2fn_refs,
        ['ann0', 'ann1', 'ann2', 'ann3'],
        imageid2fn_predictions[sys_suffix],
        imageids_gold
    )
    
    results = score_masks_macro(data_jacc, data_dice)

    # Calculate the score considering 1 gold standard -- the majority vote by pixel
    addmajoritvote_mask(imageid2fn_refs, ['ann0', 'ann1', 'ann2', 'ann3'], imageids_gold)
    results_mv = score_masks(imageid2fn_refs['majorityvote'], imageid2fn_predictions[sys_suffix], imageids_gold)

    results['jaccard'] = results_mv['jaccard']
    results['dice'] = results_mv['dice']
    results['number_segmentation_instances'] = len(imageids_gold)

    return results


if __name__ == "__main__":
    print("Starting segmentation scoring script with RGB mask handling")
    
    masks_reference_dir = sys.argv[1] if len(sys.argv) > 1 else '/app/input/ref/masks_refs'
    masks_prediction_dir = sys.argv[2] if len(sys.argv) > 2 else '/app/input/res/masks_preds'
    score_dir = sys.argv[3] if len(sys.argv) > 3 else '/app/output/'
    sys_suffix = sys.argv[4] if len(sys.argv) > 4 else 'sys'

    print(f"Reference masks directory: {masks_reference_dir}")
    print(f"Prediction masks directory: {masks_prediction_dir}")
    print(f"Output directory: {score_dir}")
    print(f"System suffix: {sys_suffix}")

    results = main(masks_reference_dir, masks_prediction_dir, sys_suffix)

    output_file = f'{score_dir}/scores_segmentation.json'
    print(f"Writing results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print("Done!")