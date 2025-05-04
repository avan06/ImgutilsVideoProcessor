import io
import os
import gc
import re
import cv2
import time
import zipfile
import tempfile
import traceback
import numpy as np
import gradio as gr
import imgutils.detect.person as person_detector
import imgutils.detect.halfbody as halfbody_detector
import imgutils.detect.head as head_detector
import imgutils.detect.face as face_detector
import imgutils.metrics.ccip as ccip_analyzer
import imgutils.metrics.dbaesthetic as dbaesthetic_analyzer
import imgutils.metrics.lpips as lpips_module
from PIL import Image
from typing import List, Tuple, Dict, Any, Union, Optional, Iterator

# --- Constants for File Types ---
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif', '.gif')
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm', '.mpeg', '.mpg')

# --- Helper Functions ---
def sanitize_filename(filename: str, max_len: int = 50) -> str:
    """Removes invalid characters and shortens a filename for safe use."""
    # Remove path components
    base_name = os.path.basename(filename)
    # Remove extension
    name_part, _ = os.path.splitext(base_name)
    # Replace spaces and problematic characters with underscores
    sanitized = re.sub(r'[\\/*?:"<>|\s]+', '_', name_part)
    # Remove leading/trailing underscores/periods
    sanitized = sanitized.strip('._')
    # Limit length (important for temp paths and OS limits)
    sanitized = sanitized[:max_len]
    # Ensure it's not empty after sanitization
    if not sanitized:
        return "file"
    return sanitized

def convert_to_pil(frame: np.ndarray) -> Image.Image:
    """Converts an OpenCV frame (BGR) to a PIL Image (RGB)."""
    # Add error handling for potentially empty frames
    if frame is None or frame.size == 0:
        raise ValueError("Cannot convert empty frame to PIL Image")
    try:
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    except Exception as e:
        # Re-raise with more context if conversion fails
        raise RuntimeError(f"Failed to convert frame to PIL Image: {e}")

def image_to_bytes(img: Image.Image, format: str = 'PNG') -> bytes:
    """Converts a PIL Image to bytes."""
    if img is None:
        raise ValueError("Cannot convert None image to bytes")
    byte_arr = io.BytesIO()
    img.save(byte_arr, format=format)
    return byte_arr.getvalue()

def create_zip_file(image_data: Dict[str, bytes], output_path: str) -> None:
    """
    Creates a zip file containing the provided images directly at the output_path.

    Args:
        image_data: A dictionary where keys are filenames (including paths within zip)
                    and values are image bytes.
        output_path: The full path where the zip file should be created.
    """
    if not image_data:
        raise ValueError("No image data provided to create zip file.")
    if not output_path:
        raise ValueError("No output path provided for the zip file.")

    print(f"Creating zip file at: {output_path}")

    try:
        # Ensure parent directory exists (useful if output_path is nested)
        # Though NamedTemporaryFile usually handles this for its own path.
        parent_dir = os.path.dirname(output_path)
        if parent_dir: # Check if there is a parent directory component
            os.makedirs(parent_dir, exist_ok=True)

        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Sort items for potentially better organization and predictability
            for filename, img_bytes in sorted(image_data.items()):
                zipf.writestr(filename, img_bytes)
        print(f"Successfully created zip file with {len(image_data)} items at {output_path}.")
        # No return value needed as we are writing to a path
    except Exception as e:
        print(f"Error creating zip file at {output_path}: {e}")
        # If zip creation fails, attempt to remove the partially created file
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                print(f"Removed partially created/failed zip file: {output_path}")
            except OSError as remove_err:
                print(f"Warning: Could not remove failed zip file {output_path}: {remove_err}")
        raise # Re-raise the original exception

def generate_filename(
    base_name: str, # Should be the core identifier, e.g., "frame_X_person_Y_scoreZ"
    aesthetic_label: Optional[str] = None,
    ccip_cluster_id_for_lpips_logic: Optional[int] = None, # Original CCIP ID, used to decide if LPIPS is sub-cluster
    ccip_folder_naming_index: Optional[int] = None,      # The new 000, 001, ... index based on image count
    source_prefix_for_ccip_folder: Optional[str] = None,  # The source filename prefix for CCIP folder
    lpips_folder_naming_index: Optional[Union[int, str]] = None, # New: Can be int (0,1,2...) or "noise"
    file_extension: str = '.png',
    # Suffix flags for this specific image:
    is_halfbody_primary_target_type: bool = False, # If this image itself was a halfbody primary target
    is_derived_head_crop: bool = False,
    is_derived_face_crop: bool = False,
) -> str:
    """
    Generates the final filename, incorporating aesthetic label, cluster directory,
    and crop indicators. CCIP and LPIPS folder names are sorted by image count.
    """
    filename_stem = base_name
    
    # Add suffixes for derived crops.
    # For halfbody primary targets, the base_name should already contain "halfbody".
    # This flag is more for potentially adding a suffix if desired, but currently not used to add a suffix.
    # if is_halfbody_primary_target_type:
    #     filename_stem += "_halfbody" # Potentially redundant if base_name good.

    if is_derived_head_crop:
        filename_stem += "_headCrop"
    if is_derived_face_crop:
        filename_stem += "_faceCrop"

    filename_with_extension = filename_stem + file_extension

    path_parts = []
    # New CCIP folder naming based on source prefix and sorted index
    if ccip_folder_naming_index is not None and source_prefix_for_ccip_folder is not None:
        path_parts.append(f"{source_prefix_for_ccip_folder}_ccip_{ccip_folder_naming_index:03d}")

    # LPIPS folder naming based on the new sorted index or "noise"
    if lpips_folder_naming_index is not None:
        lpips_folder_name_part_str: Optional[str] = None
        if isinstance(lpips_folder_naming_index, str) and lpips_folder_naming_index == "noise":
            lpips_folder_name_part_str = "noise"
        elif isinstance(lpips_folder_naming_index, int):
            lpips_folder_name_part_str = f"{lpips_folder_naming_index:03d}"
        
        if lpips_folder_name_part_str is not None:
            # Determine prefix based on whether the item was originally in a CCIP cluster
            if ccip_cluster_id_for_lpips_logic is not None: # LPIPS is sub-cluster if item had an original CCIP ID
                lpips_folder_name_base = "lpips_sub_"
            else: # No CCIP, LPIPS is primary
                lpips_folder_name_base = "lpips_"
            path_parts.append(f"{lpips_folder_name_base}{lpips_folder_name_part_str}")

    final_filename_part = filename_with_extension
    if aesthetic_label:
        final_filename_part = f"{aesthetic_label}_{filename_with_extension}"

    if path_parts:
        return f"{'/'.join(path_parts)}/{final_filename_part}"
    else:
        return final_filename_part

# --- Core Processing Function for a single source (video or image sequence) ---
def _process_input_source_frames(
    source_file_prefix: str, # Sanitized name for this source (e.g., "myvideo" or "ImageGroup123")
    # Iterator yielding: (PIL.Image, frame_identifier_string, current_item_index, total_items_for_desc)
    # For videos, current_item_index is the 1-based raw frame number.
    # For images, current_item_index is the 1-based image number in the sequence.
    frames_provider: Iterator[Tuple[Image.Image, int, int, int]],
    is_video_source: bool, # To adjust some logging/stats messages
    # Person Detection
    enable_person_detection: bool,
    min_target_width_person_percentage: float,
    person_model_name: str,
    person_conf_threshold: float,
    person_iou_threshold: float,
    # Half-Body Detection
    enable_halfbody_detection: bool,
    enable_halfbody_cropping: bool,
    min_target_width_halfbody_percentage: float,
    halfbody_model_name: str,
    halfbody_conf_threshold: float,
    halfbody_iou_threshold: float,
    # Head Detection
    enable_head_detection: bool,
    enable_head_cropping: bool,
    min_crop_width_head_percentage: float,
    enable_head_filtering: bool,
    head_model_name: str,
    head_conf_threshold: float,
    head_iou_threshold: float,
    # Face Detection
    enable_face_detection: bool,
    enable_face_cropping: bool,
    min_crop_width_face_percentage: float,
    enable_face_filtering: bool,
    face_model_name: str,
    face_conf_threshold: float,
    face_iou_threshold: float,
    # CCIP Classification
    enable_ccip_classification: bool,
    ccip_model_name: str,
    ccip_threshold: float,
    # LPIPS Clustering
    enable_lpips_clustering: bool,
    lpips_threshold: float,
    # Aesthetic Analysis
    enable_aesthetic_analysis: bool,
    aesthetic_model_name: str,
    # Gradio Progress (specific to this source's processing)
    progress_updater # Function: (progress_value: float, desc: str) -> None
) -> Tuple[str | None, str]:
    """
    Processes frames from a given source (video or image sequence) according to the specified parameters.
    Order: Person => Half-Body (alternative) => Face Detection => Head Detection => CCIP => Aesthetic.

    Returns:
        A tuple containing:
        - Path to the output zip file (or None if error).
        - Status message string.
    """
    # This list will hold data for images that pass all filters, BEFORE LPIPS and final zipping
    images_pending_final_processing: List[Dict[str, Any]] = []
    
    # CCIP specific data
    ccip_clusters_info: List[Tuple[int, np.ndarray]] = []
    next_ccip_cluster_id = 0
    
    # Stats
    processed_items_count = 0
    total_persons_detected_raw, total_halfbodies_detected_raw = 0, 0
    person_targets_processed_count, halfbody_targets_processed_count, fullframe_targets_processed_count = 0, 0, 0
    total_faces_detected_on_targets, total_heads_detected_on_targets = 0, 0
    
    # These count items added to images_pending_final_processing
    main_targets_pending_count, face_crops_pending_count, head_crops_pending_count = 0, 0, 0
    items_filtered_by_face_count, items_filtered_by_head_count = 0, 0
    ccip_applied_count, aesthetic_applied_count = 0, 0
    # LPIPS stats
    lpips_images_subject_to_clustering, total_lpips_clusters_created, total_lpips_noise_samples = 0, 0, 0

    gc_interval = 100 # items from provider
    start_time = time.time()
    
    # Progress update for initializing this specific video
    progress_updater(0, desc=f"Initializing {source_file_prefix}...")
    output_zip_path_temp = None
    output_zip_path_final = None

    try:
        # --- Main Loop for processing items from the frames_provider ---
        for pil_image_full_frame, frame_specific_index, current_item_index, total_items_for_desc in frames_provider:
            progress_value_for_updater = (current_item_index) / total_items_for_desc if total_items_for_desc > 0 else 1.0 


            # The description string should reflect what current_item_index means
            item_description = ""
            if is_video_source:
                # For video, total_items_in_source_for_description is total raw frames.
                # current_item_index is the raw frame index of the *sampled* frame.
                # We also need a counter for *sampled* frames for a "processed X of Y (sampled)" message.
                # processed_items_count counts sampled frames.
                item_description = f"Scanning frame {current_item_index}/{total_items_for_desc} (processed {processed_items_count + 1} sampled)"

            else: # For images
                item_description = f"image {current_item_index}/{total_items_for_desc}"

            progress_updater(
                min(progress_value_for_updater, 1.0), # Cap progress at 1.0
                desc=f"Processing {item_description} for {source_file_prefix}"
            )
            # processed_items_count still counts how many items are yielded by the provider
            # (i.e., how many sampled frames for video, or how many images for image sequence)
            processed_items_count += 1

            try:
                full_frame_width = pil_image_full_frame.width # Store for percentage calculations
                print(f"--- Processing item ID {frame_specific_index} (Width: {full_frame_width}px) for {source_file_prefix} ---")
                
                # List to hold PIL images that are the primary subjects for this frame
                # Each element: {'pil': Image, 'base_name': str, 'source_type': 'person'/'halfbody'/'fullframe'}
                primary_targets_for_frame: List[Dict[str, Any]] = []
                processed_primary_source_this_frame = False # Flag if Person or HalfBody yielded targets

                # --- 1. Person Detection ---
                if enable_person_detection and full_frame_width > 0:
                    print("  Attempting Person Detection...")
                    min_person_target_px_width = full_frame_width * min_target_width_person_percentage
                    person_detections = person_detector.detect_person(
                        pil_image_full_frame, model_name=person_model_name,
                        conf_threshold=person_conf_threshold, iou_threshold=person_iou_threshold
                    )
                    total_persons_detected_raw += len(person_detections)
                    if person_detections:
                        print(f"    Detected {len(person_detections)} raw persons.")
                        valid_person_targets = 0
                        for i, (bbox, _, score) in enumerate(person_detections):
                                # Check width before full crop for minor optimization
                            detected_person_width = bbox[2] - bbox[0]
                            if detected_person_width >= min_person_target_px_width:
                                primary_targets_for_frame.append({
                                    'pil': pil_image_full_frame.crop(bbox),
                                    'base_name': f"{source_file_prefix}_item_{frame_specific_index}_person_{i}_score{int(score*100)}",
                                    'source_type': 'person'})
                                person_targets_processed_count +=1
                                valid_person_targets +=1
                            else:
                                print(f"      Person {i} width {detected_person_width}px < min {min_person_target_px_width:.0f}px. Skipping.")
                        if valid_person_targets > 0:
                            processed_primary_source_this_frame = True
                            print(f"    Added {valid_person_targets} persons as primary targets.")
                
                # --- 2. Half-Body Detection (if Person not processed and HBD enabled) ---
                if not processed_primary_source_this_frame and enable_halfbody_detection and full_frame_width > 0:
                    print("  Attempting Half-Body Detection (on full item)...")
                    min_halfbody_target_px_width = full_frame_width * min_target_width_halfbody_percentage
                    halfbody_detections = halfbody_detector.detect_halfbody(
                        pil_image_full_frame, model_name=halfbody_model_name,
                        conf_threshold=halfbody_conf_threshold, iou_threshold=halfbody_iou_threshold
                    )
                    total_halfbodies_detected_raw += len(halfbody_detections)
                    if halfbody_detections:
                        print(f"    Detected {len(halfbody_detections)} raw half-bodies.")
                        valid_halfbody_targets = 0
                        for i, (bbox, _, score) in enumerate(halfbody_detections):
                            detected_hb_width = bbox[2] - bbox[0]
                            # Cropping must be enabled and width must be sufficient for it to be a target
                            if enable_halfbody_cropping and detected_hb_width >= min_halfbody_target_px_width:
                                primary_targets_for_frame.append({
                                    'pil': pil_image_full_frame.crop(bbox),
                                    'base_name': f"{source_file_prefix}_item_{frame_specific_index}_halfbody_{i}_score{int(score*100)}",
                                    'source_type': 'halfbody'})
                                halfbody_targets_processed_count +=1
                                valid_halfbody_targets +=1
                            elif enable_halfbody_cropping:
                                print(f"      Half-body {i} width {detected_hb_width}px < min {min_halfbody_target_px_width:.0f}px. Skipping.")
                        if valid_halfbody_targets > 0:
                            processed_primary_source_this_frame = True
                            print(f"    Added {valid_halfbody_targets} half-bodies as primary targets.")

                # --- 3. Full Frame/Image (fallback) ---
                if not processed_primary_source_this_frame:
                    print("  Processing Full Item as primary target.")
                    primary_targets_for_frame.append({
                        'pil': pil_image_full_frame.copy(),
                        'base_name': f"{source_file_prefix}_item_{frame_specific_index}_full",
                        'source_type': 'fullframe'})
                    fullframe_targets_processed_count += 1
                
                # --- Process each identified primary_target_for_frame ---
                for target_data in primary_targets_for_frame:
                    current_pil: Image.Image = target_data['pil']
                    current_base_name: str = target_data['base_name'] # Base name for this main target
                    current_source_type: str = target_data['source_type']
                    current_pil_width = current_pil.width # For sub-crop percentage calculations
                    print(f"    Processing target: {current_base_name} (type: {current_source_type}, width: {current_pil_width}px)")
                    
                    # Store PILs of successful crops from current_pil for this target
                    keep_this_target = True
                    item_area = current_pil_width * current_pil.height
                    potential_face_crops_pil: List[Image.Image] = []
                    potential_head_crops_pil: List[Image.Image] = []
                    
                    # --- A. Face Detection ---
                    if keep_this_target and enable_face_detection and current_pil_width > 0:
                        print(f"      Detecting faces in {current_base_name}...")
                        min_face_crop_px_width = current_pil_width * min_crop_width_face_percentage
                        face_detections = face_detector.detect_faces(
                            current_pil, model_name=face_model_name,
                            conf_threshold=face_conf_threshold, iou_threshold=face_iou_threshold
                        )
                        total_faces_detected_on_targets += len(face_detections)
                        if not face_detections and enable_face_filtering:
                            keep_this_target = False
                            items_filtered_by_face_count += 1
                            print(f"        FILTERING TARGET {current_base_name} (no face).")
                        elif face_detections and enable_face_cropping:
                            for f_idx, (f_bbox, _, _) in enumerate(face_detections):
                                if (f_bbox[2]-f_bbox[0]) >= min_face_crop_px_width:
                                    potential_face_crops_pil.append(current_pil.crop(f_bbox))
                                else:
                                    print(f"          Face {f_idx} too small. Skipping crop.")
                    
                    # --- B. Head Detection ---
                    if keep_this_target and enable_head_detection and current_pil_width > 0:
                        print(f"      Detecting heads in {current_base_name}...")
                        min_head_crop_px_width = current_pil_width * min_crop_width_head_percentage
                        head_detections = head_detector.detect_heads(
                            current_pil, model_name=head_model_name,
                            conf_threshold=head_conf_threshold, iou_threshold=head_iou_threshold
                        )
                        total_heads_detected_on_targets += len(head_detections)
                        if not head_detections and enable_head_filtering:
                            keep_this_target = False
                            items_filtered_by_head_count += 1
                            print(f"        FILTERING TARGET {current_base_name} (no head).")
                            potential_face_crops_pil.clear() # Clear faces if head filter removed target
                        elif head_detections and enable_head_cropping:
                            for h_idx, (h_bbox, _, _) in enumerate(head_detections):
                                h_w = h_bbox[2]-h_bbox[0] # h_h = h_bbox[3]-h_bbox[1]
                                if h_w >= min_head_crop_px_width and item_area > 0:
                                    potential_head_crops_pil.append(current_pil.crop(h_bbox))
                                else:
                                    print(f"          Head {h_idx} too small or too large relative to parent. Skipping crop.")
                    
                    # --- If target is filtered, clean up and skip to next target ---
                    if not keep_this_target:
                        print(f"    Target {current_base_name} was filtered by face/head presence rules. Discarding it and its potential crops.")
                        if current_pil is not None:
                            del current_pil
                        potential_face_crops_pil.clear()
                        potential_head_crops_pil.clear()
                        continue # To the next primary_target_for_frame
                    
                    # --- C. CCIP Classification (on current_pil, if it's kept) ---
                    assigned_ccip_id = None # This is the original CCIP ID
                    if enable_ccip_classification:
                        print(f"      Classifying {current_base_name} with CCIP...")
                        try:
                            feature = ccip_analyzer.ccip_extract_feature(current_pil, model=ccip_model_name)
                            best_match_cid = None
                            min_diff = float('inf')
                            # Find the best potential match among existing clusters
                            if ccip_clusters_info: # Only loop if there are clusters to compare against
                                for cid, rep_f in ccip_clusters_info:
                                    diff = ccip_analyzer.ccip_difference(feature, rep_f, model=ccip_model_name)
                                    if diff < min_diff:
                                        min_diff = diff
                                        best_match_cid = cid

                            # Decide whether to use the best match or create a new cluster
                            if best_match_cid is not None and min_diff < ccip_threshold:
                                assigned_ccip_id = best_match_cid
                                print(f"        -> Matched Cluster {assigned_ccip_id} (Diff: {min_diff:.6f} <= Threshold {ccip_threshold:.3f})")
                            else:
                                # No suitable match found (either no clusters existed, or the best match's diff was strictly greater than threshold)
                                # Create a new cluster
                                assigned_ccip_id = next_ccip_cluster_id
                                ccip_clusters_info.append((assigned_ccip_id, feature))
                                if not ccip_clusters_info or len(ccip_clusters_info) == 1:
                                    print(f"        -> New Cluster {assigned_ccip_id} (First item or no prior suitable clusters)")
                                else:
                                    # MODIFIED: Log message reflecting that new cluster is formed if diff > threshold
                                    print(f"        -> New Cluster {assigned_ccip_id} (Min diff to others: {min_diff:.6f} > Threshold {ccip_threshold:.3f})")
                                next_ccip_cluster_id += 1
                            print(f"      CCIP: Target {current_base_name} -> Original Cluster ID {assigned_ccip_id}")
                            del feature
                            ccip_applied_count += 1
                        except Exception as e_ccip:
                            print(f"      Error CCIP: {e_ccip}")
                    
                    # --- D. Aesthetic Analysis (on current_pil, if it's kept) ---
                    item_aesthetic_label = None
                    if enable_aesthetic_analysis:
                        print(f"      Analyzing {current_base_name} for aesthetics...")
                        try:
                            res = dbaesthetic_analyzer.anime_dbaesthetic(current_pil, model_name=aesthetic_model_name)
                            if isinstance(res, tuple) and len(res) >= 1:
                                item_aesthetic_label = res[0]
                            print(f"      Aesthetic: Target {current_base_name} -> {item_aesthetic_label}")
                            aesthetic_applied_count += 1
                        except Exception as e_aes:
                            print(f"      Error Aesthetic: {e_aes}")

                    add_current_pil_to_pending_list = True
                    if current_source_type == 'fullframe':
                        can_skip_fullframe_target = False
                        if enable_face_detection or enable_head_detection:
                            found_valid_sub_crop_from_enabled_detector = False
                            if enable_face_detection and len(potential_face_crops_pil) > 0:
                                found_valid_sub_crop_from_enabled_detector = True

                            if not found_valid_sub_crop_from_enabled_detector and \
                               enable_head_detection and len(potential_head_crops_pil) > 0:
                                found_valid_sub_crop_from_enabled_detector = True

                            if not found_valid_sub_crop_from_enabled_detector: # No valid crops from any enabled sub-detector
                                can_skip_fullframe_target = True # All enabled sub-detectors failed

                        if can_skip_fullframe_target:
                            add_current_pil_to_pending_list = False
                            print(f"      Skipping save of fullframe target '{current_base_name}' because all enabled sub-detectors (Face/Head) yielded no valid-width crops.")
                    
                    if add_current_pil_to_pending_list:
                        # --- E. Save current_pil (if it passed all filters) ---
                        # Add main target to pending list
                        images_pending_final_processing.append({
                            'pil_image': current_pil.copy(), 'base_name_for_filename': current_base_name,
                            'ccip_cluster_id': assigned_ccip_id, 'aesthetic_label': item_aesthetic_label,
                            'is_halfbody_primary_target_type': (current_source_type == 'halfbody'),
                            'is_derived_head_crop': False, 'is_derived_face_crop': False,
                            'lpips_cluster_id': None, # Will be filled by LPIPS clustering
                            'lpips_folder_naming_index': None # Will be filled by LPIPS renaming
                        })
                        main_targets_pending_count +=1
                    
                    # --- F. Save Face Crops (derived from current_pil) ---
                    for i, fc_pil in enumerate(potential_face_crops_pil):
                        images_pending_final_processing.append({
                            'pil_image': fc_pil, 'base_name_for_filename': f"{current_base_name}_face{i}",
                            'ccip_cluster_id': assigned_ccip_id, 'aesthetic_label': item_aesthetic_label,
                            'is_halfbody_primary_target_type': False,
                            'is_derived_head_crop': False, 'is_derived_face_crop': True,
                            'lpips_cluster_id': None,
                            'lpips_folder_naming_index': None
                        })
                        face_crops_pending_count+=1
                    potential_face_crops_pil.clear()
                    
                    # --- G. Save Head Crops (derived from current_pil) ---
                    for i, hc_pil in enumerate(potential_head_crops_pil):
                        images_pending_final_processing.append({
                            'pil_image': hc_pil, 'base_name_for_filename': f"{current_base_name}_head{i}",
                            'ccip_cluster_id': assigned_ccip_id, 'aesthetic_label': item_aesthetic_label,
                            'is_halfbody_primary_target_type': False,
                            'is_derived_head_crop': True, 'is_derived_face_crop': False,
                            'lpips_cluster_id': None,
                            'lpips_folder_naming_index': None
                        })
                        head_crops_pending_count+=1
                    potential_head_crops_pil.clear()

                    if current_pil is not None: # Ensure current_pil exists before attempting to delete
                        del current_pil # Clean up the PIL for this target_data
                
                primary_targets_for_frame.clear()
            except Exception as item_proc_err:
                print(f"!! Major Error processing item ID {frame_specific_index} for {source_file_prefix}: {item_proc_err}")
                traceback.print_exc()
                # Cleanup local vars for this item if error
                if 'primary_targets_for_frame' in locals():
                    primary_targets_for_frame.clear()
                # Also ensure current_pil from inner loop is cleaned up if error happened mid-loop
                if 'current_pil' in locals() and current_pil is not None:
                    del current_pil

            if processed_items_count % gc_interval == 0:
                gc.collect()
                print(f"  [GC triggered at {processed_items_count} items for {source_file_prefix}]")
        # --- End of Main Item Processing Loop ---
        
        print(f"\nRunning final GC before LPIPS/Zipping for {source_file_prefix}...")
        gc.collect()

        if not images_pending_final_processing:
            status_message = f"Processing for {source_file_prefix} finished, but no images were generated or passed filters for LPIPS/Zipping."
            print(status_message)
            return None, status_message
        
        # --- LPIPS Clustering Stage ---
        print(f"\n--- LPIPS Clustering Stage for {source_file_prefix} (Images pending: {len(images_pending_final_processing)}) ---")
        if enable_lpips_clustering:
            print(f"  LPIPS Clustering enabled with threshold: {lpips_threshold}")
            lpips_images_subject_to_clustering = len(images_pending_final_processing)

            if enable_ccip_classification and next_ccip_cluster_id > 0: # CCIP was used
                print("  LPIPS clustering within CCIP clusters.")
                images_by_ccip: Dict[Optional[int], List[int]] = {} # ccip_id -> list of original indices
                for i, item_data in enumerate(images_pending_final_processing):
                    ccip_id = item_data['ccip_cluster_id'] # Original CCIP ID
                    if ccip_id not in images_by_ccip:
                        images_by_ccip[ccip_id] = []
                    images_by_ccip[ccip_id].append(i)

                for ccip_id, indices_in_ccip_cluster in images_by_ccip.items():
                    pils_for_lpips_sub_cluster = [images_pending_final_processing[idx]['pil_image'] for idx in indices_in_ccip_cluster]
                    if len(pils_for_lpips_sub_cluster) > 1:
                        print(f"    Clustering {len(pils_for_lpips_sub_cluster)} images in CCIP cluster {ccip_id}...")
                        try:
                            lpips_sub_ids = lpips_module.lpips_clustering(pils_for_lpips_sub_cluster, threshold=lpips_threshold)
                            for i_sub, lpips_id in enumerate(lpips_sub_ids):
                                original_idx = indices_in_ccip_cluster[i_sub]
                                images_pending_final_processing[original_idx]['lpips_cluster_id'] = lpips_id
                        except Exception as e_lpips_sub:
                            print(f"      Error LPIPS sub-cluster CCIP {ccip_id}: {e_lpips_sub}")
                    elif len(pils_for_lpips_sub_cluster) == 1:
                         images_pending_final_processing[indices_in_ccip_cluster[0]]['lpips_cluster_id'] = 0 # type: ignore
                del images_by_ccip
                if 'pils_for_lpips_sub_cluster' in locals():
                    del pils_for_lpips_sub_cluster # Ensure cleanup
            else: # LPIPS on all images globally
                print("  LPIPS clustering on all collected images.")
                all_pils_for_global_lpips = [item['pil_image'] for item in images_pending_final_processing]
                if len(all_pils_for_global_lpips) > 1:
                    try:
                        lpips_global_ids = lpips_module.lpips_clustering(all_pils_for_global_lpips, threshold=lpips_threshold)
                        for i, lpips_id in enumerate(lpips_global_ids):
                            images_pending_final_processing[i]['lpips_cluster_id'] = lpips_id
                    except Exception as e_lpips_global:
                        print(f"      Error LPIPS global: {e_lpips_global}")
                elif len(all_pils_for_global_lpips) == 1:
                    images_pending_final_processing[0]['lpips_cluster_id'] = 0 # type: ignore
                del all_pils_for_global_lpips
                
            # Calculate LPIPS stats
            all_final_lpips_ids = [item.get('lpips_cluster_id') for item in images_pending_final_processing if item.get('lpips_cluster_id') is not None]
            if all_final_lpips_ids:
                unique_lpips_clusters = set(filter(lambda x: x != -1, all_final_lpips_ids))
                total_lpips_clusters_created = len(unique_lpips_clusters)
                total_lpips_noise_samples = sum(1 for x in all_final_lpips_ids if x == -1)
        else:
            print("  LPIPS Clustering disabled.")

        # --- CCIP Folder Renaming Logic ---
        original_ccip_id_to_new_naming_index: Dict[int, int] = {}
        if enable_ccip_classification:
            print(f"  Preparing CCIP folder renaming for {source_file_prefix}...")
            ccip_image_counts: Dict[int, int] = {} # original_ccip_id -> count of images in it
            for item_data_for_count in images_pending_final_processing:
                original_ccip_id_val = item_data_for_count.get('ccip_cluster_id')
                if original_ccip_id_val is not None:
                    ccip_image_counts[original_ccip_id_val] = ccip_image_counts.get(original_ccip_id_val, 0) + 1

            if ccip_image_counts:
                # Sort original ccip_ids by their counts in descending order
                sorted_ccip_groups_by_count: List[Tuple[int, int]] = sorted(
                    ccip_image_counts.items(),
                    key=lambda item: item[1],  # Sort by count
                    reverse=True
                )
                for new_idx, (original_id, count) in enumerate(sorted_ccip_groups_by_count):
                    original_ccip_id_to_new_naming_index[original_id] = new_idx
                    print(f"    CCIP Remap for {source_file_prefix}: Original ID {original_id} (count: {count}) -> New Naming Index {new_idx:03d}")
            else:
                print(f"    No CCIP-assigned images found for {source_file_prefix} to perform renaming.")

        # --- LPIPS Folder Renaming Logic ---
        if enable_lpips_clustering:
            print(f"  Preparing LPIPS folder renaming for {source_file_prefix}...")
            # Initialize/Reset lpips_folder_naming_index for all items
            for item_data in images_pending_final_processing:
                item_data['lpips_folder_naming_index'] = None

            if enable_ccip_classification and next_ccip_cluster_id > 0: # LPIPS within CCIP
                print(f"    LPIPS renaming within CCIP clusters for {source_file_prefix}.")
                items_grouped_by_original_ccip: Dict[Optional[int], List[Dict[str, Any]]] = {}
                for item_data in images_pending_final_processing:
                    original_ccip_id = item_data.get('ccip_cluster_id')
                    if original_ccip_id not in items_grouped_by_original_ccip: items_grouped_by_original_ccip[original_ccip_id] = []
                    items_grouped_by_original_ccip[original_ccip_id].append(item_data)

                for original_ccip_id, items_in_ccip in items_grouped_by_original_ccip.items():
                    lpips_counts_in_ccip: Dict[int, int] = {} # original_lpips_id (non-noise) -> count
                    for item_data in items_in_ccip:
                        lpips_id = item_data.get('lpips_cluster_id')
                        if lpips_id is not None and lpips_id != -1:
                            lpips_counts_in_ccip[lpips_id] = lpips_counts_in_ccip.get(lpips_id, 0) + 1
                    
                    lpips_id_to_naming_in_ccip: Dict[int, Union[int, str]] = {}
                    if lpips_counts_in_ccip:
                        sorted_lpips = sorted(lpips_counts_in_ccip.items(), key=lambda x: x[1], reverse=True)
                        for new_idx, (lpips_id, count) in enumerate(sorted_lpips):
                            lpips_id_to_naming_in_ccip[lpips_id] = new_idx
                            ccip_disp = f"OrigCCIP-{original_ccip_id}" if original_ccip_id is not None else "NoCCIP"
                            print(f"      LPIPS Remap in {ccip_disp}: OrigLPIPS ID {lpips_id} (count: {count}) -> New Naming Index {new_idx:03d}")
                    
                    for item_data in items_in_ccip:
                        lpips_id = item_data.get('lpips_cluster_id')
                        if lpips_id is not None:
                            if lpips_id == -1: item_data['lpips_folder_naming_index'] = "noise"
                            elif lpips_id in lpips_id_to_naming_in_ccip:
                                item_data['lpips_folder_naming_index'] = lpips_id_to_naming_in_ccip[lpips_id]
                del items_grouped_by_original_ccip 
            else: # Global LPIPS
                print(f"    Global LPIPS renaming for {source_file_prefix}.")
                global_lpips_counts: Dict[int, int] = {}
                for item_data in images_pending_final_processing:
                    lpips_id = item_data.get('lpips_cluster_id')
                    if lpips_id is not None and lpips_id != -1:
                        global_lpips_counts[lpips_id] = global_lpips_counts.get(lpips_id, 0) + 1

                global_lpips_id_to_naming: Dict[int, Union[int, str]] = {}
                if global_lpips_counts:
                    sorted_global_lpips = sorted(global_lpips_counts.items(), key=lambda x: x[1], reverse=True)
                    for new_idx, (lpips_id, count) in enumerate(sorted_global_lpips):
                        global_lpips_id_to_naming[lpips_id] = new_idx
                        print(f"      Global LPIPS Remap: OrigLPIPS ID {lpips_id} (count: {count}) -> New Naming Index {new_idx:03d}")

                for item_data in images_pending_final_processing:
                    lpips_id = item_data.get('lpips_cluster_id')
                    if lpips_id is not None:
                        if lpips_id == -1: item_data['lpips_folder_naming_index'] = "noise"
                        elif lpips_id in global_lpips_id_to_naming:
                            item_data['lpips_folder_naming_index'] = global_lpips_id_to_naming[lpips_id]
            gc.collect()

        # --- Final Zipping Stage ---
        images_to_zip: Dict[str, bytes] = {}
        print(f"\n--- Final Zipping Stage for {source_file_prefix} ({len(images_pending_final_processing)} items) ---")
        for item_data in images_pending_final_processing:
            original_ccip_id_for_item = item_data.get('ccip_cluster_id')
            current_ccip_naming_idx_for_folder: Optional[int] = None

            if enable_ccip_classification and original_ccip_id_for_item is not None and \
               original_ccip_id_for_item in original_ccip_id_to_new_naming_index:
                current_ccip_naming_idx_for_folder = original_ccip_id_to_new_naming_index[original_ccip_id_for_item]

            current_lpips_naming_idx_for_folder = item_data.get('lpips_folder_naming_index')

            final_filename = generate_filename(
                base_name=item_data['base_name_for_filename'],
                aesthetic_label=item_data.get('aesthetic_label'),
                ccip_cluster_id_for_lpips_logic=original_ccip_id_for_item,
                ccip_folder_naming_index=current_ccip_naming_idx_for_folder,
                source_prefix_for_ccip_folder=source_file_prefix if current_ccip_naming_idx_for_folder is not None else None,
                lpips_folder_naming_index=current_lpips_naming_idx_for_folder,
                is_halfbody_primary_target_type=item_data['is_halfbody_primary_target_type'],
                is_derived_head_crop=item_data['is_derived_head_crop'],
                is_derived_face_crop=item_data['is_derived_face_crop']
            )
            try:
                images_to_zip[final_filename] = image_to_bytes(item_data['pil_image'])
            except Exception as e_bytes:
                print(f"  Error converting/adding {final_filename} to zip: {e_bytes}")
            finally:
                if 'pil_image' in item_data and item_data['pil_image'] is not None:
                    del item_data['pil_image']
        images_pending_final_processing.clear()

        if not images_to_zip:
            status_message = f"Processing for {source_file_prefix} finished, but no images were converted for zipping."
            print(status_message)
            return None, status_message

        print(f"Preparing zip file for {source_file_prefix} with {len(images_to_zip)} images...")
        progress_updater(1.0, desc=f"Creating Zip File for {source_file_prefix}...")
        zip_start_time = time.time()
        
        # Use NamedTemporaryFile with delete=False for the final output path
        # This file will persist until manually cleaned or OS cleanup
        temp_zip_file = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        output_zip_path_temp = temp_zip_file.name
        temp_zip_file.close() # Close the handle, but file remains

        try:
            # Write data to the temporary file path
            create_zip_file(images_to_zip, output_zip_path_temp)
            zip_duration = time.time() - zip_start_time
            print(f"Temporary zip file for {source_file_prefix} created in {zip_duration:.2f} seconds at {output_zip_path_temp}")
            
            # Construct the new, desired filename
            temp_dir = os.path.dirname(output_zip_path_temp)
            timestamp = int(time.time())
            desired_filename = f"{source_file_prefix}_processed_{timestamp}.zip"
            output_zip_path_final = os.path.join(temp_dir, desired_filename)
            
            # Rename the temporary file to the desired name
            print(f"Renaming temp file for {source_file_prefix} to: {output_zip_path_final}")
            os.rename(output_zip_path_temp, output_zip_path_final)
            print("Rename successful.")
            output_zip_path_temp = None # Clear temp path as it's been renamed

        except Exception as zip_or_rename_err:
            print(f"Error during zip creation or renaming for {source_file_prefix}: {zip_or_rename_err}")
            # Clean up the *original* temp file if it still exists and renaming failed
            if output_zip_path_temp and os.path.exists(output_zip_path_temp):
                try:
                    os.remove(output_zip_path_temp)
                except OSError:
                    pass
            if output_zip_path_final and os.path.exists(output_zip_path_final): # Check if rename partially happened
                try:
                    os.remove(output_zip_path_final)
                except OSError:
                    pass
            raise zip_or_rename_err # Re-raise the error
        
        # --- Prepare Status Message ---
        processing_duration = time.time() - start_time - zip_duration # Exclude zipping time from processing time
        total_duration = time.time() - start_time # Includes zipping/renaming
        
        # --- Build final status message ---
        person_stats = "N/A"
        if enable_person_detection:
            person_stats = f"{total_persons_detected_raw} raw, {person_targets_processed_count} targets (>{min_target_width_person_percentage*100:.1f}% itemW)"

        halfbody_stats = "N/A"
        if enable_halfbody_detection:
            halfbody_stats = f"{total_halfbodies_detected_raw} raw, {halfbody_targets_processed_count} targets (>{min_target_width_halfbody_percentage*100:.1f}% itemW)"
        fullframe_stats = f"{fullframe_targets_processed_count} targets"

        face_stats = "N/A"
        if enable_face_detection:
            face_stats = f"{total_faces_detected_on_targets} on targets, {face_crops_pending_count} crops pending (>{min_crop_width_face_percentage*100:.1f}% parentW)"
            if enable_face_filtering:
                face_stats += f", {items_filtered_by_face_count} targets filtered"

        head_stats = "N/A"
        if enable_head_detection:
            head_stats = f"{total_heads_detected_on_targets} on targets, {head_crops_pending_count} crops pending (>{min_crop_width_head_percentage*100:.1f}% parentW)"
            if enable_head_filtering:
                head_stats += f", {items_filtered_by_head_count} targets filtered"

        ccip_stats = "N/A"
        if enable_ccip_classification:
            ccip_stats = f"{next_ccip_cluster_id} original clusters created, on {ccip_applied_count} targets. Folders renamed by image count."

        lpips_stats = "N/A"
        if enable_lpips_clustering:
            lpips_stats = f"{lpips_images_subject_to_clustering} images processed, {total_lpips_clusters_created} clusters, {total_lpips_noise_samples} noise. Folders renamed by image count."

        aesthetic_stats = "N/A"
        if enable_aesthetic_analysis:
            aesthetic_stats = f"On {aesthetic_applied_count} targets"

        item_desc_for_stats = "Items from Provider" if not is_video_source else "Sampled Frames"
        status_message = (
            f"Processing for '{source_file_prefix}' Complete!\n"
            f"Total time: {total_duration:.2f}s (Proc: {processing_duration:.2f}s, Zip: {zip_duration:.2f}s)\n"
            f"{item_desc_for_stats}: {total_items_for_desc}, Processed Items: {processed_items_count}\n"
            f"--- Primary Targets Processed ---\n"
            f"  Person Detection: {person_stats}\n"
            f"  Half-Body Detection: {halfbody_stats}\n"
            f"  Full Item Processing: {fullframe_stats}\n"
            f"--- Items Pending Final Processing ({main_targets_pending_count} main, {face_crops_pending_count} face, {head_crops_pending_count} head) ---\n"
            f"  Face Detection: {face_stats}\n"
            f"  Head Detection: {head_stats}\n"
            f"  CCIP Classification: {ccip_stats}\n"
            f"  LPIPS Clustering: {lpips_stats}\n"
            f"  Aesthetic Analysis: {aesthetic_stats}\n"
            f"Zip file contains {len(images_to_zip)} images.\n"
            f"Output Zip: {output_zip_path_final}"
        )
        print(status_message)
        progress_updater(1.0, desc=f"Finished {source_file_prefix}!")
            
        # Return the path to the zip file
        return output_zip_path_final, status_message

    except Exception as e:
        print(f"!! An unhandled error occurred during processing of {source_file_prefix}: {e}")
        traceback.print_exc() # Print detailed traceback for debugging
        # Clean up main data structures
        images_pending_final_processing.clear()
        ccip_clusters_info.clear()
        gc.collect()
        
        # Clean up temp file if it exists on general error
        if output_zip_path_temp and os.path.exists(output_zip_path_temp):
            try:
                os.remove(output_zip_path_temp)
            except OSError:
                pass
        
        # Clean up final file if it exists on general error (maybe renaming succeeded but later code failed)
        if output_zip_path_final and os.path.exists(output_zip_path_final):
            try:
                os.remove(output_zip_path_final)
            except OSError:
                pass
        return None, f"An error occurred with {source_file_prefix}: {e}"

# --- Main Processing Function for Input files ---
def process_inputs_main(
    input_file_objects: List[Any], # Gradio File component gives list of tempfile._TemporaryFileWrapper
    sample_interval_ms: int, # Relevant for videos only
    # Person Detection
    enable_person_detection: bool,
    min_target_width_person_percentage: float,
    person_model_name: str,
    person_conf_threshold: float,
    person_iou_threshold: float,
    # Half-Body Detection
    enable_halfbody_detection: bool,
    enable_halfbody_cropping: bool,
    min_target_width_halfbody_percentage: float,
    halfbody_model_name: str,
    halfbody_conf_threshold: float,
    halfbody_iou_threshold: float,
    # Head Detection
    enable_head_detection: bool,
    enable_head_cropping: bool,
    min_crop_width_head_percentage: float,
    enable_head_filtering: bool,
    head_model_name: str,
    head_conf_threshold: float,
    head_iou_threshold: float,
    # Face Detection
    enable_face_detection: bool,
    enable_face_cropping: bool,
    min_crop_width_face_percentage: float,
    enable_face_filtering: bool,
    face_model_name: str,
    face_conf_threshold: float,
    face_iou_threshold: float,
    # CCIP Classification
    enable_ccip_classification: bool,
    ccip_model_name: str,
    ccip_threshold: float,
    # LPIPS Clustering
    enable_lpips_clustering: bool,
    lpips_threshold: float,
    # Aesthetic Analysis
    enable_aesthetic_analysis: bool,
    aesthetic_model_name: str,
    progress=gr.Progress(track_tqdm=True) # Gradio progress for overall processing
) -> Tuple[Optional[List[str]], str]: # Returns list of ZIP paths and combined status

    if not input_file_objects:
        return [], "Error: No files provided."

    video_file_temp_objects: List[Any] = []
    image_file_temp_objects: List[Any] = []

    for file_obj in input_file_objects:
        # gr.Files returns a list of tempfile._TemporaryFileWrapper objects
        # We need the .name attribute to get the actual file path
        file_name = getattr(file_obj, 'orig_name', file_obj.name) # Use original name if available
        if isinstance(file_name, str):
            lower_file_name = file_name.lower()
            if any(lower_file_name.endswith(ext) for ext in VIDEO_EXTENSIONS):
                video_file_temp_objects.append(file_obj)
            elif any(lower_file_name.endswith(ext) for ext in IMAGE_EXTENSIONS):
                image_file_temp_objects.append(file_obj)
            else:
                print(f"Warning: File '{file_name}' has an unrecognized extension and will be skipped.")
        else:
            print(f"Warning: File object {file_obj} does not have a valid name and will be skipped.")


    output_zip_paths_all_sources = []
    all_status_messages = []
    
    total_processing_tasks = (1 if image_file_temp_objects else 0) + len(video_file_temp_objects)
    if total_processing_tasks == 0:
        return [], "No processable video or image files found in the input."
        
    tasks_completed_count = 0
    
    # Print overall settings once
    print(f"--- Overall Batch Processing Settings ---")
    print(f"  Number of image sequences to process: {1 if image_file_temp_objects else 0}")
    print(f"  Number of videos to process: {len(video_file_temp_objects)}")
    print(f"  Sample Interval (for videos): {sample_interval_ms}ms")
    print(f"  Detection Order: Person => Half-Body (alt) => Face => Head. Then: CCIP => LPIPS => Aesthetic.")
    print(f"  Person Detect = {enable_person_detection}" + (f" (MinW:{min_target_width_person_percentage*100:.1f}%, Mdl:{person_model_name}, Conf:{person_conf_threshold:.2f}, IoU:{person_iou_threshold:.2f})" if enable_person_detection else ""))
    print(f"  HalfBody Detect = {enable_halfbody_detection}" + (f" (FullFrameOnly, Crop:{enable_halfbody_cropping}, MinW:{min_target_width_halfbody_percentage*100:.1f}%, Mdl:{halfbody_model_name}, Conf:{halfbody_conf_threshold:.2f}, IoU:{halfbody_iou_threshold:.2f})" if enable_halfbody_detection else ""))
    print(f"  Face Detect = {enable_face_detection}" + (f" (Crop:{enable_face_cropping}, MinW:{min_crop_width_face_percentage*100:.1f}%, Filter:{enable_face_filtering}, Mdl:{face_model_name}, Conf:{face_conf_threshold:.2f}, IoU:{face_iou_threshold:.2f})" if enable_face_detection else ""))
    print(f"  Head Detect = {enable_head_detection}" + (f" (Crop:{enable_head_cropping}, MinW:{min_crop_width_head_percentage*100:.1f}%, Filter:{enable_head_filtering}, Mdl:{head_model_name}, Conf:{head_conf_threshold:.2f}, IoU:{head_iou_threshold:.2f})" if enable_head_detection else ""))
    print(f"  CCIP Classify = {enable_ccip_classification}" + (f" (Mdl:{ccip_model_name}, Thr:{ccip_threshold:.3f})" if enable_ccip_classification else ""))
    print(f"  LPIPS Clustering = {enable_lpips_clustering}" + (f" (Thr:{lpips_threshold:.3f})" if enable_lpips_clustering else ""))
    print(f"  Aesthetic Analyze = {enable_aesthetic_analysis}" + (f" (Mdl:{aesthetic_model_name})" if enable_aesthetic_analysis else ""))
    print(f"--- End of Overall Settings ---")


    # --- Process Image Sequence (if any) ---
    if image_file_temp_objects:
        image_group_label_base = "ImageGroup"
        # Attempt to use first image name for more uniqueness, fallback to timestamp
        try:
            first_image_orig_name = getattr(image_file_temp_objects[0], 'orig_name', image_file_temp_objects[0].name)
            image_group_label_base = sanitize_filename(first_image_orig_name, max_len=20)
        except:
            pass # Stick with "ImageGroup"
        
        image_source_file_prefix = f"{image_group_label_base}_{int(time.time())}"
        
        current_task_number = tasks_completed_count + 1
        progress_description_prefix = f"Image Seq. {current_task_number}/{total_processing_tasks} ({image_source_file_prefix})"
        progress(tasks_completed_count / total_processing_tasks, desc=f"{progress_description_prefix}: Starting...")
        print(f"\n>>> Processing Image Sequence: {image_source_file_prefix} ({len(image_file_temp_objects)} images) <<<")

        def image_frames_provider_generator() -> Iterator[Tuple[Image.Image, int, int, int]]:
            num_images = len(image_file_temp_objects)
            for idx, img_obj in enumerate(image_file_temp_objects):
                try:
                    pil_img = Image.open(img_obj.name).convert('RGB')
                    yield pil_img, idx, idx + 1, num_images
                except Exception as e_load:
                    print(f"Error loading image {getattr(img_obj, 'orig_name', img_obj.name)}: {e_load}. Skipping.")
                    # If we skip, the total_items_in_source for _process_input_source_frames might be off
                    # For simplicity, we'll proceed, but this could be refined to adjust total_items dynamically.
                    # Or, pre-filter loadable images. For now, just skip.
                    continue
        
        def image_group_progress_updater(item_progress_value: float, desc: str):
            overall_progress = (tasks_completed_count + item_progress_value) / total_processing_tasks
            progress(overall_progress, desc=f"{progress_description_prefix}: {desc}")

        try:
            zip_file_path_single, status_message_single = _process_input_source_frames(
                source_file_prefix=image_source_file_prefix,
                frames_provider=image_frames_provider_generator(),
                is_video_source=False,
                enable_person_detection=enable_person_detection,
                min_target_width_person_percentage=min_target_width_person_percentage,
                person_model_name=person_model_name,
                person_conf_threshold=person_conf_threshold,
                person_iou_threshold=person_iou_threshold,
                enable_halfbody_detection=enable_halfbody_detection,
                enable_halfbody_cropping=enable_halfbody_cropping,
                min_target_width_halfbody_percentage=min_target_width_halfbody_percentage,
                halfbody_model_name=halfbody_model_name,
                halfbody_conf_threshold=halfbody_conf_threshold,
                halfbody_iou_threshold=halfbody_iou_threshold,
                enable_head_detection=enable_head_detection,
                enable_head_cropping=enable_head_cropping,
                min_crop_width_head_percentage=min_crop_width_head_percentage,
                enable_head_filtering=enable_head_filtering,
                head_model_name=head_model_name,
                head_conf_threshold=head_conf_threshold,
                head_iou_threshold=head_iou_threshold,
                enable_face_detection=enable_face_detection,
                enable_face_cropping=enable_face_cropping,
                min_crop_width_face_percentage=min_crop_width_face_percentage,
                enable_face_filtering=enable_face_filtering,
                face_model_name=face_model_name,
                face_conf_threshold=face_conf_threshold,
                face_iou_threshold=face_iou_threshold,
                enable_ccip_classification=enable_ccip_classification,
                ccip_model_name=ccip_model_name,
                ccip_threshold=ccip_threshold,
                enable_lpips_clustering=enable_lpips_clustering,
                lpips_threshold=lpips_threshold,
                enable_aesthetic_analysis=enable_aesthetic_analysis,
                aesthetic_model_name=aesthetic_model_name,
                progress_updater=image_group_progress_updater
            )
            if zip_file_path_single:
                output_zip_paths_all_sources.append(zip_file_path_single)
                all_status_messages.append(f"--- Image Sequence ({image_source_file_prefix}) Processing Succeeded ---\n{status_message_single}")
            else:
                all_status_messages.append(f"--- Image Sequence ({image_source_file_prefix}) Processing Failed ---\n{status_message_single}")
        except Exception as e_img_seq:
            error_msg = f"Critical error during processing of image sequence {image_source_file_prefix}: {e_img_seq}"
            print(error_msg)
            traceback.print_exc()
            all_status_messages.append(f"--- Image Sequence ({image_source_file_prefix}) Processing CRITICALLY FAILED ---\n{error_msg}")
        
        tasks_completed_count += 1
        print(f">>> Finished attempt for Image Sequence: {image_source_file_prefix} <<<")

    # --- Process Video Files (if any) ---
    for video_idx, video_file_temp_obj in enumerate(video_file_temp_objects):
        video_path_temp = video_file_temp_obj.name
        video_original_filename = os.path.basename(getattr(video_file_temp_obj, 'orig_name', video_path_temp))
        video_source_file_prefix = sanitize_filename(video_original_filename)

        current_task_number = tasks_completed_count + 1
        progress_description_prefix = f"Video {current_task_number}/{total_processing_tasks}"
        
        print(f"\n>>> Processing Video: {video_original_filename} (Sanitized Prefix: {video_source_file_prefix}) <<<")
        progress(tasks_completed_count / total_processing_tasks, desc=f"{progress_description_prefix}: Starting processing...")

        # It yields: (PIL.Image, frame_identifier_string, current_raw_frame_index_from_video, total_items_for_desc)
        # The third element will be the raw frame number based on CAP_PROP_POS_FRAMES or current_pos_ms
        # to align progress with total_items_for_desc (raw frame count).
        def video_frames_provider_generator(video_path: str, interval_ms: int) -> Iterator[Tuple[Image.Image, int, int, int]]:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video file for provider: {video_path}")
                return

            total_items_for_desc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_items_for_desc <= 0:
                print(f"Warning: Video {video_original_filename} reported {total_items_for_desc} frames. This might be inaccurate. Proceeding...")
                # If it's 0, the progress in _process_input_source_frames might behave unexpectedly.
                # Setting to 1 to avoid division by zero, but this means progress won't be very useful.
                total_items_for_desc = 1 # Fallback to prevent division by zero

            # processed_count_in_provider = 0 # Counts *sampled* frames, not used for progress index
            last_processed_ms = -float('inf')
            raw_frames_read_by_provider = 0 # Counts all frames read by cap.read()

            try:
                while True:
                    # For progress, use current_pos_ms or CAP_PROP_POS_FRAMES
                    # CAP_PROP_POS_FRAMES is a 0-based index of the next frame to be decoded/captured.
                    current_raw_frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) # Use this for progress
                    current_pos_ms_in_provider = cap.get(cv2.CAP_PROP_POS_MSEC)
                    
                    # Loop break condition (more robust)
                    if raw_frames_read_by_provider > 0 and current_pos_ms_in_provider <= last_processed_ms and interval_ms > 0 :
                         # If interval_ms is 0 or very small, current_pos_ms might not advance much for consecutive reads.
                         # Adding a check for raw_frames_read_by_provider against a large number or CAP_PROP_FRAME_COUNT
                         # could be an additional safety, but CAP_PROP_FRAME_COUNT can be unreliable.
                         # The ret_frame check is the primary exit.
                        pass # Let ret_frame handle the actual end. This check is for stuck videos.


                    should_process_this_frame = current_pos_ms_in_provider >= last_processed_ms + interval_ms - 1
                    
                    ret_frame, frame_cv_data = cap.read()
                    if not ret_frame: # Primary exit point for the loop
                        break 
                    raw_frames_read_by_provider +=1 # Incremented after successful read

                    if should_process_this_frame:
                        try:
                            pil_img = convert_to_pil(frame_cv_data)
                            last_processed_ms = current_pos_ms_in_provider
                            yield pil_img, int(current_pos_ms_in_provider), current_raw_frame_index + 1, total_items_for_desc # Yield 1-based raw frame index
                        except Exception as e_conv:
                            print(f"Error converting frame at {current_pos_ms_in_provider}ms (raw index {current_raw_frame_index}) for {video_original_filename}: {e_conv}. Skipping.")
                        finally:
                            pass 
            finally:
                if cap.isOpened():
                    cap.release()
                    print(f"   Video capture for provider ({video_original_filename}) released.")
        
        def video_progress_updater(item_progress_value: float, desc: str):
            overall_progress = (tasks_completed_count + item_progress_value) / total_processing_tasks
            progress(overall_progress, desc=f"{progress_description_prefix}: {desc}")

        try:
            zip_file_path_single, status_message_single = _process_input_source_frames(
                source_file_prefix=video_source_file_prefix,
                frames_provider=video_frames_provider_generator(video_path_temp, sample_interval_ms),
                is_video_source=True,
                enable_person_detection=enable_person_detection,
                min_target_width_person_percentage=min_target_width_person_percentage,
                person_model_name=person_model_name,
                person_conf_threshold=person_conf_threshold,
                person_iou_threshold=person_iou_threshold,
                enable_halfbody_detection=enable_halfbody_detection,
                enable_halfbody_cropping=enable_halfbody_cropping,
                min_target_width_halfbody_percentage=min_target_width_halfbody_percentage,
                halfbody_model_name=halfbody_model_name,
                halfbody_conf_threshold=halfbody_conf_threshold,
                halfbody_iou_threshold=halfbody_iou_threshold,
                enable_head_detection=enable_head_detection,
                enable_head_cropping=enable_head_cropping,
                min_crop_width_head_percentage=min_crop_width_head_percentage,
                enable_head_filtering=enable_head_filtering,
                head_model_name=head_model_name,
                head_conf_threshold=head_conf_threshold,
                head_iou_threshold=head_iou_threshold,
                enable_face_detection=enable_face_detection,
                enable_face_cropping=enable_face_cropping,
                min_crop_width_face_percentage=min_crop_width_face_percentage,
                enable_face_filtering=enable_face_filtering,
                face_model_name=face_model_name,
                face_conf_threshold=face_conf_threshold,
                face_iou_threshold=face_iou_threshold,
                enable_ccip_classification=enable_ccip_classification,
                ccip_model_name=ccip_model_name,
                ccip_threshold=ccip_threshold,
                enable_lpips_clustering=enable_lpips_clustering,
                lpips_threshold=lpips_threshold,
                enable_aesthetic_analysis=enable_aesthetic_analysis,
                aesthetic_model_name=aesthetic_model_name,
                progress_updater=video_progress_updater
            )
            if zip_file_path_single:
                output_zip_paths_all_sources.append(zip_file_path_single)
                all_status_messages.append(f"--- Video ({video_original_filename}) Processing Succeeded ---\n{status_message_single}")
            else:
                all_status_messages.append(f"--- Video ({video_original_filename}) Processing Failed ---\n{status_message_single}")

        except Exception as e_vid:
            # This catches errors if process_video itself raises an unhandled exception
            # (though process_video has its own try-except)
            error_msg = f"Critical error during processing of video {video_original_filename}: {e_vid}"
            print(error_msg)
            traceback.print_exc()
            all_status_messages.append(f"--- Video ({video_original_filename}) Processing CRITICALLY FAILED ---\n{error_msg}")

        tasks_completed_count += 1
        print(f">>> Finished attempt for Video: {video_original_filename} <<<")
        # Gradio manages the lifecycle of video_path_temp (the uploaded temp file)

    final_summary_message = "\n\n==============================\n\n".join(all_status_messages)
    
    successful_zips_count = len(output_zip_paths_all_sources)
    if successful_zips_count == 0 and total_processing_tasks > 0:
        final_summary_message = f"ALL {total_processing_tasks} INPUT SOURCE(S) FAILED TO PRODUCE A ZIP FILE.\n\n" + final_summary_message
    elif total_processing_tasks > 0:
        final_summary_message = f"Successfully processed {successful_zips_count} out of {total_processing_tasks} input source(s).\n\n" + final_summary_message
    else: # Should be caught earlier by "No processable files"
        final_summary_message = "No inputs were processed."

    progress(1.0, desc="All processing attempts finished.")
    
    # gr.Files output expects a list of file paths. An empty list is fine if no files.
    return output_zip_paths_all_sources, final_summary_message


# --- Gradio Interface Setup ---

css = """
/* Default (Light Mode) Styles */
#warning {
    background-color: #FFCCCB; /* Light red background */
    padding: 10px;
    border-radius: 5px;
    color: #A00000;        /* Dark red text */
    border: 1px solid #E5B8B7; /* A slightly darker border for more definition */
}
/* Dark Mode Styles */
@media (prefers-color-scheme: dark) {
    #warning {
        background-color: #5C1A1A; /* Darker red background, suitable for dark mode */
        color: #FFDDDD;        /* Light pink text, for good contrast against the dark red background */
        border: 1px solid #8B0000; /* A more prominent dark red border in dark mode */
    }
}
#status_box {
    white-space: pre-wrap !important; /* Ensure status messages show newlines */
    font-family: monospace; /* Optional: Use monospace for better alignment */
}
"""

# --- Define Model Lists ---
person_models = ['person_detect_v1.3_s', 'person_detect_v1.2_s', 'person_detect_v1.1_s', 'person_detect_v1.1_m', 'person_detect_v1_m', 'person_detect_v1.1_n', 'person_detect_v0_s', 'person_detect_v0_m', 'person_detect_v0_x']
halfbody_models = ['halfbody_detect_v1.0_s', 'halfbody_detect_v1.0_n', 'halfbody_detect_v0.4_s', 'halfbody_detect_v0.3_s', 'halfbody_detect_v0.2_s']
head_models = ['head_detect_v2.0_s', 'head_detect_v2.0_m', 'head_detect_v2.0_n', 'head_detect_v2.0_x', 'head_detect_v2.0_s_yv11', 'head_detect_v2.0_m_yv11', 'head_detect_v2.0_n_yv11', 'head_detect_v2.0_x_yv11', 'head_detect_v2.0_l_yv11']
face_models = ['face_detect_v1.4_s', 'face_detect_v1.4_n', 'face_detect_v1.3_s', 'face_detect_v1.3_n', 'face_detect_v1.2_s', 'face_detect_v1.1_s', 'face_detect_v1.1_n', 'face_detect_v1_s', 'face_detect_v1_n', 'face_detect_v0_s', 'face_detect_v0_n']
ccip_models = ['ccip-caformer-24-randaug-pruned', 'ccip-caformer-6-randaug-pruned_fp32', 'ccip-caformer-5_fp32']
aesthetic_models = ['swinv2pv3_v0_448_ls0.2_x', 'swinv2pv3_v0_448_ls0.2', 'caformer_s36_v0_ls0.2']

with gr.Blocks(css=css) as demo:
    gr.Markdown("# Video Processor using dghs-imgutils")
    gr.Markdown("Upload one or more videos, or a sequence of images. Videos are processed individually, while multiple images are treated as a single sequence. Each processed source (video or image sequence) is then sequentially analyzed by [dghs-imgutils](https://github.com/deepghs/imgutils) to detect subjects, classify items, and process its content according to your settings, ultimately generating a ZIP file with the extracted images.")
    gr.Markdown("**Detection Flow:** " + 
                "[Person](https://dghs-imgutils.deepghs.org/main/api_doc/detect/person.html) ⇒ " +
                "[Half-Body](https://dghs-imgutils.deepghs.org/main/api_doc/detect/halfbody.html) (if no person) ⇒ " +
                "[Face](https://dghs-imgutils.deepghs.org/main/api_doc/detect/face.html) (on target) ⇒ " + 
                "[Head](https://dghs-imgutils.deepghs.org/main/api_doc/detect/head.html) (on target).")
    gr.Markdown("**Analysis Flow:** " + 
                "[CCIP](https://dghs-imgutils.deepghs.org/main/api_doc/metrics/ccip.html) Clustering ⇒ " + 
                "[LPIPS](https://dghs-imgutils.deepghs.org/main/api_doc/metrics/lpips.html) Clustering ⇒ " + 
                "[Aesthetic](https://dghs-imgutils.deepghs.org/main/api_doc/metrics/dbaesthetic.html) Labeling.")
    gr.Markdown("**Note on CCIP Folders:** CCIP cluster folders are named `{source_prefix}_ccip_XXX`, sorted by image count (most images = `_ccip_000`).")
    gr.Markdown("**Note on LPIPS Folders:** LPIPS cluster folders (e.g., `lpips_XXX` or `lpips_sub_XXX`) are also sorted by image count within their scope. 'noise' folders are named explicitly.")

    with gr.Row():
        with gr.Column(scale=1):
            # --- Input Components ---
            process_button = gr.Button("Process Input(s) & Generate ZIP(s)", variant="primary")
            input_files = gr.Files(label="Upload Videos or Image Sequences", file_types=['video', 'image'], file_count="multiple") 
            sample_interval_ms = gr.Number(label="Sample Interval (ms, for videos)", value=1000, minimum=1, step=100)
            
            # --- Detection Options ---
            gr.Markdown("**Detection Options**")
            # --- Person Detection Block ---
            with gr.Accordion("Person Detection Options", open=True):
                enable_person_detection = gr.Checkbox(label="Enable Person Detection", value=True)
                with gr.Group() as person_detection_params_group:
                    min_target_width_person_percentage_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.25, step=0.01,
                        label="Min Target Width (% of Item Width)",
                        info="Minimum width for a detected person to be processed (e.g., 0.25 = 25%)."
                    )
                    person_model_name_dd = gr.Dropdown(person_models, label="PD Model", value=person_models[0])
                    person_conf_threshold = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="PD Conf")
                    person_iou_threshold = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="PD IoU")
                enable_person_detection.change(fn=lambda e: gr.update(visible=e), inputs=enable_person_detection, outputs=person_detection_params_group)
                
            # --- Half-Body Detection Block ---
            with gr.Accordion("Half-Body Detection Options", open=True):
                enable_halfbody_detection = gr.Checkbox(label="Enable Half-Body Detection", value=True)
                with gr.Group() as halfbody_params_group:
                    gr.Markdown("<small>_Detects half-bodies in full items if Person Detection is off/fails._</small>")
                    enable_halfbody_cropping = gr.Checkbox(label="Use Half-Bodies as Targets", value=True)
                    min_target_width_halfbody_percentage_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.25, step=0.01,
                        label="Min Target Width (% of Item Width)",
                        info="Minimum width for a detected half-body to be processed (e.g., 0.25 = 25%)."
                    )
                    halfbody_model_name_dd = gr.Dropdown(halfbody_models, label="HBD Model", value=halfbody_models[0])
                    halfbody_conf_threshold = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="HBD Conf")
                    halfbody_iou_threshold = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="HBD IoU")
                enable_halfbody_detection.change(fn=lambda e: gr.update(visible=e), inputs=enable_halfbody_detection, outputs=halfbody_params_group)
                
            # --- Face Detection Block ---
            with gr.Accordion("Face Detection Options", open=True):
                enable_face_detection = gr.Checkbox(label="Enable Face Detection", value=True)
                with gr.Group() as face_params_group:
                    enable_face_filtering = gr.Checkbox(label="Filter Targets Without Detected Faces", value=True)
                    enable_face_cropping = gr.Checkbox(label="Crop Detected Faces", value=False)
                    min_crop_width_face_percentage_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.2, step=0.01,
                        label="Min Crop Width (% of Parent Width)",
                        info="Minimum width for a face crop relative to its parent image's width (e.g., 0.2 = 20%)."
                    )
                    face_model_name_dd = gr.Dropdown(face_models, label="FD Model", value=face_models[0])
                    face_conf_threshold = gr.Slider(0.0, 1.0, value=0.25, step=0.05, label="FD Conf")
                    face_iou_threshold = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="FD IoU")
                enable_face_detection.change(fn=lambda e: gr.update(visible=e), inputs=enable_face_detection, outputs=face_params_group)
                
            # --- Head Detection Block ---
            with gr.Accordion("Head Detection Options", open=True):
                enable_head_detection = gr.Checkbox(label="Enable Head Detection", value=True)
                with gr.Group() as head_params_group:
                    gr.Markdown("<small>_Detects heads in targets. Crops if meets width req._</small>")
                    enable_head_filtering = gr.Checkbox(label="Filter Targets Without Heads", value=True)
                    enable_head_cropping = gr.Checkbox(label="Crop Detected Heads", value=False)
                    min_crop_width_head_percentage_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.2, step=0.01,
                        label="Min Crop Width (% of Parent Width)",
                        info="Minimum width for a head crop relative to its parent image's width (e.g., 0.2 = 20%)."
                    )
                    head_model_name_dd = gr.Dropdown(head_models, label="HD Model", value=head_models[0])
                    head_conf_threshold = gr.Slider(0.0, 1.0, value=0.4, step=0.05, label="HD Conf")
                    head_iou_threshold = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="HD IoU")
                enable_head_detection.change(fn=lambda e: gr.update(visible=e), inputs=enable_head_detection, outputs=head_params_group)
            
            # --- Analysis/Classification Options ---
            gr.Markdown("**Analysis & Classification**")
            # --- CCIP Classification Block ---
            with gr.Accordion("CCIP Classification Options", open=True):
                 enable_ccip_classification = gr.Checkbox(label="Enable CCIP Classification", value=True)
                 with gr.Group() as ccip_params_group:
                    gr.Markdown("<small>_Clusters results by similarity. Folders sorted by image count._</small>")
                    ccip_model_name_dd = gr.Dropdown(ccip_models, label="CCIP Model", value=ccip_models[0])
                    ccip_threshold_slider = gr.Slider(0.0, 1.0, step=0.01, value=0.20, label="CCIP Similarity Threshold")
                 enable_ccip_classification.change(fn=lambda e: gr.update(visible=e), inputs=enable_ccip_classification, outputs=ccip_params_group)

            # LPIPS Clustering Options
            with gr.Accordion("LPIPS Clustering Options", open=True):
                enable_lpips_clustering = gr.Checkbox(label="Enable LPIPS Clustering", value=True)
                with gr.Group() as lpips_params_group: 
                    gr.Markdown("<small>_Clusters images by LPIPS similarity. Applied after CCIP (if enabled) or globally. Folders sorted by image count._</small>")
                    lpips_threshold_slider = gr.Slider(0.0, 1.0, step=0.01, value=0.45, label="LPIPS Similarity Threshold")
                enable_lpips_clustering.change(fn=lambda e: gr.update(visible=e), inputs=enable_lpips_clustering, outputs=lpips_params_group)
            
            # --- Aesthetic Analysis Block ---
            with gr.Accordion("Aesthetic Analysis Options", open=True):
                 enable_aesthetic_analysis = gr.Checkbox(label="Enable Aesthetic Analysis (Anime)", value=True)
                 with gr.Group() as aesthetic_params_group:
                     gr.Markdown("<small>_Prepends aesthetic label to filenames._</small>")
                     aesthetic_model_name_dd = gr.Dropdown(aesthetic_models, label="Aesthetic Model", value=aesthetic_models[0])
                 enable_aesthetic_analysis.change(fn=lambda e: gr.update(visible=e), inputs=enable_aesthetic_analysis, outputs=aesthetic_params_group)

            gr.Markdown("---")
            gr.Markdown("**Warning:** Complex combinations can be slow. Models downloaded on first use.", elem_id="warning")

        with gr.Column(scale=1):
            # --- Output Components ---
            status_text = gr.Textbox(label="Processing Status", interactive=False, lines=20, elem_id="status_box")
            output_zips = gr.Files(label="Download Processed Images (ZIPs)") 
            
    # Connect button click
    process_button.click(
        fn=process_inputs_main,
        inputs=[
            input_files, sample_interval_ms,
            # Person Detect
            enable_person_detection, min_target_width_person_percentage_slider,
            person_model_name_dd, person_conf_threshold, person_iou_threshold,
            # HalfBody Detect
            enable_halfbody_detection, enable_halfbody_cropping, min_target_width_halfbody_percentage_slider,
            halfbody_model_name_dd, halfbody_conf_threshold, halfbody_iou_threshold,
            # Head Detect
            enable_head_detection, enable_head_cropping, min_crop_width_head_percentage_slider,
            enable_head_filtering, head_model_name_dd, head_conf_threshold, head_iou_threshold,
            # Face Detect
            enable_face_detection, enable_face_cropping, min_crop_width_face_percentage_slider,
            enable_face_filtering, face_model_name_dd, face_conf_threshold, face_iou_threshold,
            # CCIP
            enable_ccip_classification, ccip_model_name_dd, ccip_threshold_slider,
            # LPIPS
            enable_lpips_clustering, lpips_threshold_slider,
            # Aesthetic
            enable_aesthetic_analysis, aesthetic_model_name_dd,
        ],
        outputs=[output_zips, status_text]
    )

# --- Launch Script ---
if __name__ == "__main__":
    print("Starting Gradio App...")
    # Model pre-check
    try:
        print("Checking/Downloading models (this might take a moment)...")
        # Use simple, small images for checks
        dummy_img_pil = Image.new('RGB', (64, 64), color = 'orange')
        print("  - Person detection...")
        _ = person_detector.detect_person(dummy_img_pil, model_name=person_models[0])
        print("  - HalfBody detection...")
        _ = halfbody_detector.detect_halfbody(dummy_img_pil, model_name=halfbody_models[0])
        print("  - Head detection...")
        _ = head_detector.detect_heads(dummy_img_pil, model_name=head_models[0])
        print("  - Face detection...")
        _ = face_detector.detect_faces(dummy_img_pil, model_name=face_models[0])
        print("  - CCIP feature extraction...")
        _ = ccip_analyzer.ccip_extract_feature(dummy_img_pil, size=384, model=ccip_models[0])
        print("  - LPIPS feature extraction...")
        _ = lpips_module.lpips_extract_feature(dummy_img_pil)
        print("  - Aesthetic analysis...")
        _ = dbaesthetic_analyzer.anime_dbaesthetic(dummy_img_pil, model_name=aesthetic_models[0])
        print("Models seem ready or downloaded.")
        del dummy_img_pil
        gc.collect()
    except Exception as model_err:
        print(f"\n--- !!! WARNING !!! ---")
        print(f"Could not pre-check/download all models: {model_err}")
        print(f"Models will be downloaded when first used by the application, which may cause a delay on the first run.")
        print(f"Check your internet connection and library installation (pip install \"dghs-imgutils[gpu]\").")
        print(f"-----------------------\n")
    # Launch the app
    demo.launch(inbrowser=True)
