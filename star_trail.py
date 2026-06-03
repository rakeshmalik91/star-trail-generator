# Star Trail Stacker
# ------------------
# Stacks multiple Sony ARW or standard image files into a single 'Lighten' blend stack.
# Optimized for high-resolution sequences with professional-grade stabilization.
#
# Methodology:
#   1. Ground-Stabilize: Uses bottom-masked Phase Correlation and cumulative smoothing
#      to anchor the landscape and remove high-frequency wind jitter.
#   2. Star-Stabilize: Tracks a specific celestial anchor and uses a Gaussian temporal
#      filter to iron out jitter while preserving the natural orbital trails.
#
# Usage:
#     python star_trail.py "path/to/frames" --ground-stabilize --bright 1.2
#     python star_trail.py "path/to/frames" --star-stabilize --limit 100
#     python star_trail.py "D:\Pictures\2025-09-27 - Hali Country TL2" --ground-stabilize -o "data\" --overwrite
#     python star_trail.py "D:\Pictures\2025-09-27 - Hali Country TL2" --ground-stabilize --smooth 300 --brightness-limit auto --bright 1.1 -o "data\2025-09-27 - Hali Country TL2_star_trail_ground_locked_sm300.jpg" --overwrite
#     python star_trail.py "D:\Pictures\2025-09-27 - Hali Country TL2" --ground-stabilize --video --fps 30 --video-width 3840  -o "data\" --overwrite
#     python star_trail.py "D:\Pictures\2025-09-27 - Hali Country TL1" --video no-trail --fps 30 --video-width 3840  -o "data\" --overwrite
#
# Dependencies: rawpy, numpy, Pillow, opencv-python

import os
import sys
import argparse
import time
import rawpy
from PIL import Image, ImageOps
import numpy as np
import cv2
import concurrent.futures
import subprocess
import json
from datetime import datetime

# Constants - file path & suffix
EXIFTOOL_PATH = os.path.join("lib", "exif-tools", "exiftool.exe")
SUFFIX_STAR_LOCKED = "_star_trail_star_locked"
SUFFIX_GROUND_LOCKED = "_star_trail_ground_locked"
SUFFIX_DEFAULT = "_star_trail"
SUFFIX_SMOOTH = "_sm"
SUFFIX_JPG_DIR = "_jpg"
DEFAULT_FOLDER_NAME = "stacked"
SUFFIX_STABILIZED = "_stabilized"
SUFFIX_TIMELAPSE = "_timelapse"

# Constants - input filtering
MIN_TIMESTAMP_DIFF = 0.5  # Seconds. Gaps smaller than this are treated as invalid/overlaps.
DEFAULT_BRIGHTNESS_LIMIT = "auto"  # 'auto' = use std-dev, 'none' or 1.0 = disabled.
AUTO_BRIGHTNESS_FACTOR = 0.5       # Threshold = Mean + (Factor * StdDev)

# Constants - stabilization
DEFAULT_SMOOTH_MIN = 20
DEFAULT_SMOOTH_FACTOR = 1.0  # 1.0 = 100% of sequence length
DEFAULT_MAX_ROTATION = 5.0  # Degrees. Max per-frame rotation from feature matching. 0 = disable.
DEFAULT_MOTION_WIDTH = 1500  # Target width for image downscaling during motion detection.

# Re-using common formatting from utils if available, otherwise defining here
def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0: return f"{h}h {m}m {s}s"
    if m > 0: return f"{m}m {s}s"
    return f"{s}s"

def smooth_curve(curve, radius=DEFAULT_SMOOTH_MIN):
    """
    Applies Gaussian smoothing to the motion curve to isolate jitter from natural motion.
    Detrends the curve first to prevent edge artifacts on non-stationary sweeps (e.g. drift).
    """
    if radius <= 0 or len(curve) <= radius:
        return curve
        
    # Detrend
    x_axis = np.arange(len(curve))
    coeffs = np.polyfit(x_axis, curve, 2)
    trend = np.polyval(coeffs, x_axis)
    detrended = curve - trend
    
    # Create Gaussian Kernel
    x = np.linspace(-3, 3, 2 * radius + 1)
    kernel = np.exp(-0.5 * x**2)
    kernel /= kernel.sum()
    
    # Pad edges to avoid artifacts
    padded = np.pad(detrended, (radius, radius), mode='edge')
    smooth = np.convolve(padded, kernel, mode='same')
    
    # Retrend
    return smooth[radius:-radius] + trend

def get_exif_timestamps(src_dir, files):
    """
    Extracts DateTimeOriginal from images using exiftool.
    Returns a numpy array of timestamps.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    exiftool_path = os.path.join(script_dir, EXIFTOOL_PATH)
    if not os.path.exists(exiftool_path):
        exiftool_path = "exiftool" # System fallback
        
    print(f"[*] Extracting timestamps for {len(files)} files...")
    all_timestamps = {}
    
    try:
        # Process in chunks to avoid command line length limits
        for i in range(0, len(files), 200):
            chunk = files[i:i+200]
            paths = [os.path.join(src_dir, f) for f in chunk]
            cmd = [exiftool_path, "-DateTimeOriginal", "-j"] + paths
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                for entry in data:
                    fname = os.path.basename(entry.get('SourceFile', ''))
                    # Strictly use DateTimeOriginal for "Image Taken On"
                    dt_str = entry.get('DateTimeOriginal')
                    if dt_str:
                        try:
                            # Format: 2024:04:10 16:38:44
                            dt = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
                            all_timestamps[fname] = dt.timestamp()
                        except ValueError:
                            pass
    except Exception:
        pass
        
    if not all_timestamps:
        return None
        
    # Return as array, fallback to index-based spacing ONLY if some but not all found
    ts = []
    last_t = None
    for f in files:
        t = all_timestamps.get(f)
        if t is None:
            t = (last_t + 30) if last_t is not None else 0
        ts.append(t)
        last_t = t
        
    ts = np.array(ts)
    
    # Validation: Ensure at least MIN_TIMESTAMP_DIFF between all frames
    if len(ts) > 1:
        diffs = np.diff(ts)
        if np.any(diffs < MIN_TIMESTAMP_DIFF):
            print(f"Warning: Timestamps are too close (<{MIN_TIMESTAMP_DIFF}s) or identical. Falling back to standard smoothing.")
            return None
            
    return ts

def smooth_curve_time_aware(curve, timestamps, radius_frames):
    """
    Performs Gaussian smoothing respecting the time gaps between samples.
    Detrends the curve first to prevent edge shifting artifacts when slow drift is present.
    """
    if radius_frames <= 0 or len(curve) <= 2:
        return curve
        
    # Detrend to prevent edge curling
    t_axis = timestamps - timestamps[0]
    coeffs = np.polyfit(t_axis, curve, 2)
    trend = np.polyval(coeffs, t_axis)
    detrended = curve - trend
        
    # Estimate average interval (excluding large gaps)
    intervals = np.diff(timestamps)
    med = np.median(intervals) if len(intervals) > 0 else 30
    clean_intervals = intervals[intervals < (med * 5)]
    avg_interval = np.mean(clean_intervals) if len(clean_intervals) > 0 else med
    
    sigma = radius_frames * avg_interval / 2.0
    
    smoothed = np.zeros_like(curve)
    for i in range(len(curve)):
        diff = timestamps - timestamps[i]
        # Weights based on time distance
        weights = np.exp(-0.5 * (diff / sigma)**2)
        total_weight = np.sum(weights)
        if total_weight > 0:
            smoothed[i] = np.sum(detrended * weights) / total_weight
        else:
            smoothed[i] = detrended[i]
            
    return smoothed + trend

def get_brightness(path):
    """
    Returns the mean brightness of the image (0.0 to 1.0).
    Fast develop for RAW files.
    """
    try:
        if path.lower().endswith('.arw'):
            with rawpy.imread(path) as raw:
                rgb = raw.postprocess(half_size=True, no_auto_bright=True, bright=8.0)
                return np.mean(rgb) / 255.0
        else:
            with Image.open(path) as img:
                img.thumbnail((256, 256))
                return np.mean(np.array(img)) / 255.0
    except Exception as e:
        if not hasattr(get_brightness, "warned"):
            print(f"\n[!] Error during brightness check: {e}")
            get_brightness.warned = True
        return 0.0

def get_default_output(src_dir, is_star_locked=False, is_ground_locked=False, smooth_radius=None):
    """
    Determines default output path. 
    """
    abs_src = os.path.abspath(src_dir)
    folder_name = os.path.basename(abs_src.rstrip(os.sep))
    if not folder_name:
        folder_name = DEFAULT_FOLDER_NAME
        
    if is_star_locked:
        suffix = SUFFIX_STAR_LOCKED
    elif is_ground_locked:
        suffix = SUFFIX_GROUND_LOCKED
    else:
        suffix = SUFFIX_DEFAULT
    
    if smooth_radius is not None:
        suffix += f"{SUFFIX_SMOOTH}{smooth_radius}"
        
    filename = f"{folder_name}{suffix}.jpg"
    
    parent = os.path.dirname(abs_src.rstrip(os.sep))
    if os.path.basename(parent).lower() == "data":
        return os.path.join(parent, filename)
    return filename

def get_brightest_star(gray_img):
    """
    Finds the (x, y) coordinates of the brightest point in a grayscale image.
    Uses a larger blur to find a significant star, not a hot pixel.
    """
    smooth = cv2.GaussianBlur(gray_img, (15, 15), 0)
    _, _, _, max_loc = cv2.minMaxLoc(smooth)
    return max_loc

def track_star_patch(ref_gray, curr_gray, last_pos, window_size=100):
    """
    Tracks a star patch using sub-pixel phase correlation in a local window.
    """
    x, y = int(last_pos[0]), int(last_pos[1])
    h, w = ref_gray.shape
    
    # Define window bounds
    half = window_size // 2
    x1, x2 = max(0, x - half), min(w, x + half)
    y1, y2 = max(0, y - half), min(h, y + half)
    
    # Extract patches
    ref_p = ref_gray[y1:y2, x1:x2].astype(np.float32)
    curr_p = curr_gray[y1:y2, x1:x2].astype(np.float32)
    
    if ref_p.shape != curr_p.shape or ref_p.size == 0:
        return last_pos, (0, 0)
        
    win = cv2.createHanningWindow((ref_p.shape[1], ref_p.shape[0]), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(ref_p, curr_p, win)
    
    if response > 0.1:
        new_pos = (last_pos[0] + shift[0], last_pos[1] + shift[1])
        return new_pos, shift
    return last_pos, (0, 0)

def decompose_affine(M, center=(0,0)):
    """Decomposes a 2x3 affine partial matrix into (tx, ty, angle, scale) relative to a pivot."""
    a = M[0, 0]
    b = M[1, 0]
    angle = np.arctan2(b, a)
    scale = np.sqrt(a**2 + b**2)
    
    # Translation accounting for rotation around center
    tx_rot = center[0] - a * center[0] + b * center[1]
    ty_rot = center[1] - b * center[0] - a * center[1]
    
    tx_pure = (M[0, 2] - tx_rot)
    ty_pure = (M[1, 2] - ty_rot)
    
    return tx_pure, ty_pure, angle, scale

def compose_affine(tx, ty, angle, scale=1.0, center=(0,0)):
    """Composes a 2x3 affine partial matrix from (tx, ty, angle, scale) relative to a pivot."""
    M = cv2.getRotationMatrix2D(center, np.rad2deg(angle), scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return np.float32(M)

def to_3x3(m23):
    return np.vstack([m23, [0, 0, 1]])

def from_3x3(m33):
    return m33[0:2, 0:3]

def estimate_ground_motion(ref_gray, curr_gray, y_cut_ratio=0.6, max_rotation_deg=DEFAULT_MAX_ROTATION):
    """
    Estimates ground motion (Translation + Rotation) between consecutive frames.
    Uses Phase Correlation for robust sub-pixel translation, then ORB feature
    matching with RANSAC for rotation detection (handles rotations of any size).
    Returns (dx, dy, dangle) in content-movement convention.
    """
    h, w = ref_gray.shape
    y_cut = int(h * y_cut_ratio)
    
    # Ground region: bottom portion of the image
    ref_ground = ref_gray[y_cut:, :].astype(np.float32)
    curr_ground = curr_gray[y_cut:, :].astype(np.float32)
    
    # Phase correlation for translation (proven reliable, sub-pixel)
    shift, _ = cv2.phaseCorrelate(ref_ground, curr_ground)
    dx, dy = shift[0], shift[1]
    dangle = 0.0
    
    # Feature-based rotation detection via ORB + RANSAC similarity transform
    if max_rotation_deg > 0:
        try:
            # Convert to uint8 for ORB feature detection
            ref_u8 = cv2.normalize(ref_ground, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            curr_u8 = cv2.normalize(curr_ground, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            orb = cv2.ORB_create(1000)
            kp1, des1 = orb.detectAndCompute(ref_u8, None)
            kp2, des2 = orb.detectAndCompute(curr_u8, None)
            
            if des1 is not None and des2 is not None and len(des1) >= 10 and len(des2) >= 10:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                
                if len(matches) >= 6:
                    matches = sorted(matches, key=lambda m: m.distance)
                    
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    
                    # Similarity transform (4 DOF: tx, ty, rotation, uniform scale)
                    # Maps ref points to curr points (content-movement convention)
                    M, inliers = cv2.estimateAffinePartial2D(
                        src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0
                    )
                    
                    if M is not None:
                        n_inliers = int(inliers.sum()) if inliers is not None else 0
                        if n_inliers >= 4:
                            angle = np.arctan2(M[1, 0], M[0, 0])
                            if abs(angle) <= np.deg2rad(max_rotation_deg):
                                dangle = angle
        except Exception:
            pass  # Feature matching failed, use translation-only
    
    return dx, dy, dangle


def process_single_frame(idx, filename, src_dir, export_dir, correction, bright=1.0):
    """
    Worker function. Applies sub-pixel translation or perspective correction.
    """
    path = os.path.join(src_dir, filename)
    try:
        rgb = None
        if filename.lower().endswith('.arw'):
            # ... (postprocess logic)
            with rawpy.imread(path) as raw:
                rgb = raw.postprocess(use_camera_wb=True, bright=bright, no_auto_bright=True)
            if export_dir:
                frame_jpg_path = os.path.join(export_dir, os.path.splitext(filename)[0] + ".jpg")
                Image.fromarray(rgb).save(frame_jpg_path, "JPEG", quality=95)
        else:
            with Image.open(path) as img:
                img = ImageOps.exif_transpose(img)
                if img.mode != 'RGB': img = img.convert('RGB')
                rgb = np.array(img)

        if rgb is not None:
            h_f, w_f = rgb.shape[:2]
            if correction is not None:
                if isinstance(correction, np.ndarray):
                    if correction.shape == (3, 3):
                        rgb = cv2.warpPerspective(rgb, correction, (w_f, h_f))
                    elif correction.shape == (2, 3):
                        rgb = cv2.warpAffine(rgb, correction, (w_f, h_f))
                else:
                    # Tuple or list of (dx, dy)
                    dx, dy = correction[0], correction[1]
                    M = np.float32([[1, 0, dx], [0, 1, dy]])
                    rgb = cv2.warpAffine(rgb, M, (w_f, h_f))
        return idx, rgb
    except Exception as e:
        print(f"\nError processing {filename}: {e}")
        return idx, None

def stack_star_trails(src_dir, output_path, limit=0, overwrite=False, skip=False, save_jpg=False, save_steps=False, bright=1.0, threads=None, star_stabilize=False, ground_stabilize=False, smooth_radius=None, brightness_limit=DEFAULT_BRIGHTNESS_LIMIT, max_rotation=DEFAULT_MAX_ROTATION, motion_width=DEFAULT_MOTION_WIDTH, video_mode=None, fps=30, video_width=3840):
    if not os.path.exists(src_dir):
        print(f"Error: Directory not found: {src_dir}")
        return

    # Discovery
    extensions = ('.arw', '.jpg', '.jpeg', '.png', '.tif', '.tiff')
    files = sorted([f for f in os.listdir(src_dir) if f.lower().endswith(extensions)])
    if not files:
        print(f"No compatible images found in {src_dir}")
        return

    # Prepare JPG export directory if requested
    export_dir = None
    if save_jpg:
        export_dir = os.path.abspath(src_dir).rstrip(os.sep) + SUFFIX_JPG_DIR
        os.makedirs(export_dir, exist_ok=True)
        print(f"[*] Frames will be saved to: {export_dir}")

    if limit > 0:
        files = files[:limit]

    total_files = len(files)
    
    # --- PHASE 1: Brightness Filtering ---
    if brightness_limit != 1.0:
        print(f"Phase 1: Filtering overexposed frames (mode: {brightness_limit})...")
        valid_files = []
        ignored_files = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as filter_executor:
            paths = [os.path.join(src_dir, f) for f in files]
            futures = {filter_executor.submit(get_brightness, p): f for p, f in zip(paths, files)}
            
            results_map = {}
            bright_start_time = time.time()
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                fname = futures[future]
                try:
                    b = future.result()
                    results_map[fname] = b
                except Exception:
                    results_map[fname] = 0.0
                
                elapsed = time.time() - bright_start_time
                avg = elapsed / (i + 1)
                rem = avg * (len(files) - (i + 1))
                progress_msg = f"Analysing brightness: {i+1}/{len(files)}"
                print(f"\r{progress_msg:<40} ({format_time(elapsed)} elapsed, ~{format_time(rem)} left)", end="", flush=True)
            
            print() # Clear progress line
            
            if results_map:
                all_vals = list(results_map.values())
                print(f"[*] Brightness stats: min={min(all_vals):.4f}, max={max(all_vals):.4f}, avg={np.mean(all_vals):.4f}")
            
            # Determine threshold
            thresh = brightness_limit
            if brightness_limit == 'auto' and results_map:
                all_vals = np.array(list(results_map.values()))
                # Robustly ignore frames that are 0 (failed to read) when calculating mean/std
                clean_vals = all_vals[all_vals > 0]
                if range(len(clean_vals)):
                    mean_b = np.mean(clean_vals)
                    std_b = np.std(clean_vals)
                    thresh = mean_b + (AUTO_BRIGHTNESS_FACTOR * std_b)
                    print(f"[*] Auto-threshold calculated: {thresh:.4f} (mean={mean_b:.4f}, std={std_b:.4f}, factor={AUTO_BRIGHTNESS_FACTOR})")
                else:
                    thresh = 1.0 # Fallback
            
            for f in files:
                b = results_map.get(f, 0.0)
                if b <= thresh:
                    valid_files.append(f)
                else:
                    ignored_files.append(f)
                    print(f"[-] Ignoring {f} (brightness {b:.3f} > threshold {thresh:.3f})")
        
        if len(ignored_files) > 0:
            print(f"[*] Filtered out {len(ignored_files)} frames.")
            files = valid_files
            total_files = len(files)
            if total_files == 0:
                print("Error: No files remaining after filtering.")
                return 
    else:
        print("Phase 1: Brightness Filtering [Skipped]")

    if os.path.exists(output_path):
        if overwrite:
            pass
        elif skip:
            print(f"Skipping: {output_path} already exists.")
            return
        else:
            response = input(f"[?] {output_path} already exists. Overwrite? (y/n): ").lower()
            if response != 'y':
                print("Aborting.")
                return

    total_files = len(files)
    
    # --- PHASE 2: Motion Capture ---
    corrections = [None] * total_files
    if ground_stabilize or star_stabilize:
        print(f"Phase 2: Capturing camera motion across {total_files} files...")
        timestamps = get_exif_timestamps(src_dir, files)
        
        # Step A: Capture relative frame-to-frame motion
        rel_motions = [] # List of (dx, dy, dangle) relative per-frame
        ref_plate = None
        gray_frames = []  # Save downscaled grays for validation
        
        saved_scale = None  # scale factor for full->small
        
        for idx in range(total_files):
            filename = files[idx]
            path = os.path.join(src_dir, filename)
            
            if idx == 0:
                capture_start_time = time.time()
                
            elapsed = time.time() - capture_start_time
            avg = elapsed / (idx + 1)
            rem = avg * (total_files - (idx + 1))
            progress_msg = f"Analysing jitter: {idx+1}/{total_files}"
            print(f"\r{progress_msg:<40} ({format_time(elapsed)} elapsed, ~{format_time(rem)} left)", end="", flush=True)
            
            plate = None
            if filename.lower().endswith('.arw'):
                with rawpy.imread(path) as raw:
                    plate = raw.postprocess(use_camera_wb=True, bright=8.0, no_auto_bright=True)
            else:
                with Image.open(path) as img:
                    img = ImageOps.exif_transpose(img)
                    if img.mode != 'RGB': img = img.convert('RGB')
                    plate = np.clip(np.array(img).astype(np.float32) * 3.0, 0, 255).astype(np.uint8)
            
            h_f, w_f = plate.shape[:2]
            target_w = min(w_f, motion_width) if motion_width > 0 else w_f
            scale = target_w / w_f
            if saved_scale is None:
                saved_scale = scale
            p_small = cv2.resize(plate, (target_w, int(h_f * scale)))
            gray = cv2.cvtColor(p_small, cv2.COLOR_RGB2GRAY)

            if ref_plate is None:
                ref_plate = gray
                rel_motions.append((0.0, 0.0, 0.0))
                if star_stabilize:
                    sx, sy = get_brightest_star(gray)
                    last_star_pos = (sx, sy)
            else:
                if star_stabilize:
                    last_star_pos, _ = track_star_patch(ref_plate, gray, last_star_pos)
                    rel_motions.append(last_star_pos)
                elif ground_stabilize:
                    dx, dy, dangle = estimate_ground_motion(ref_plate, gray, max_rotation_deg=max_rotation)
                    rel_motions.append((dx, dy, dangle))
                ref_plate = gray
            
            if ground_stabilize:
                gray_frames.append(gray)

        # Step B: Build cumulative motion path
        motion_path = []
        if star_stabilize:
            motion_path = rel_motions
        else:
            # Convert relative to numpy for median filtering
            rel_np = np.array(rel_motions)
            
            # 3-point Median Filter to reject sudden spikes/outliers
            clean_rel = rel_np.copy()
            if len(clean_rel) > 2:
                for i in range(1, len(clean_rel) - 1):
                    clean_rel[i] = np.median(rel_np[i-1:i+2], axis=0)
            
            # Simple cumulative sum to build absolute motion path
            cum_x, cum_y, cum_a = 0.0, 0.0, 0.0
            for i in range(len(clean_rel)):
                cum_x += clean_rel[i][0]
                cum_y += clean_rel[i][1]
                cum_a += clean_rel[i][2]
                motion_path.append((cum_x, cum_y, cum_a))

            # Diagnostic: show rotation stats
            angles_deg = np.array([p[2] for p in motion_path]) * (180.0 / np.pi)
            print(f"\n[*] Rotation stats: min={angles_deg.min():.4f}° max={angles_deg.max():.4f}° range={np.ptp(angles_deg):.4f}°")

        
        # Step C: Smoothing
        rad = smooth_radius if smooth_radius is not None else max(DEFAULT_SMOOTH_MIN, int(total_files * DEFAULT_SMOOTH_FACTOR))
        x_vals = np.array([p[0] for p in motion_path])
        y_vals = np.array([p[1] for p in motion_path])
        a_vals = np.array([p[2] for p in motion_path]) if not star_stabilize else None
        
        if timestamps is not None:
            x_guide = smooth_curve_time_aware(x_vals, timestamps, radius_frames=rad)
            y_guide = smooth_curve_time_aware(y_vals, timestamps, radius_frames=rad)
            smoothing_msg = f"Trail smoothed using time-aware radius={rad}."
        else:
            x_guide = smooth_curve(x_vals, radius=rad)
            y_guide = smooth_curve(y_vals, radius=rad)
            smoothing_msg = f"Trail smoothed using radius={rad} (no EXIF time data)."
        
        # Rotation guide: polynomial fit
        if a_vals is not None:
            if timestamps is not None:
                t_axis = (timestamps - timestamps[0])
                t_range = t_axis[-1] if t_axis[-1] > 0 else 1.0
                t_norm = t_axis / t_range
            else:
                t_norm = np.linspace(0, 1, len(a_vals))
            poly_coeffs = np.polyfit(t_norm, a_vals, deg=2)
            a_guide = np.polyval(poly_coeffs, t_norm)
        else:
            a_guide = None
        
        # Step D: Compute corrections and VALIDATE rotation
        if star_stabilize:
            for i in range(len(motion_path)):
                corrections[i] = ((x_guide[i] - x_vals[i]) / scale, (y_guide[i] - y_vals[i]) / scale)
        else:
            # Compute BOTH translation-only and translation+rotation corrections
            corr_translate = []
            corr_rotate = []
            center_full = (w_f / 2.0, h_f / 2.0)
            center_small = (gray_frames[0].shape[1] / 2.0, gray_frames[0].shape[0] / 2.0)
            
            for i in range(len(motion_path)):
                dtx = (x_guide[i] - x_vals[i]) / saved_scale
                dty = (y_guide[i] - y_vals[i]) / saved_scale
                da = a_guide[i] - a_vals[i] if a_vals is not None else 0.0
                
                corr_translate.append((dtx, dty))
                corr_rotate.append(compose_affine(dtx, dty, da, center=center_full))
            
            # Validate: measure ground alignment quality on sample frames
            # Compare SSD (Sum of Squared Differences) of edges vs center
            ref_gray = gray_frames[0]
            h_s, w_s = ref_gray.shape
            y_cut = int(h_s * 0.6)
            ref_ground = ref_gray[y_cut:, :].astype(np.float32)
            edge_w = w_s // 4  # 25% from each side
            
            sample_step = max(1, total_files // 10)
            ssd_t_edge = 0.0
            ssd_r_edge = 0.0
            ssd_t_center = 0.0
            ssd_r_center = 0.0
            n_samples = 0
            
            for idx in range(sample_step, total_files, sample_step):
                frame = gray_frames[idx]
                
                # Apply translation-only (at small scale)
                dtx_s = (x_guide[idx] - x_vals[idx])
                dty_s = (y_guide[idx] - y_vals[idx])
                M_t = np.float32([[1, 0, dtx_s], [0, 1, dty_s]])
                corrected_t = cv2.warpAffine(frame, M_t, (w_s, h_s))
                ground_t = corrected_t[y_cut:, :].astype(np.float32)
                
                # Apply translation+rotation (at small scale)
                da = a_guide[idx] - a_vals[idx] if a_vals is not None else 0.0
                M_r = compose_affine(dtx_s, dty_s, da, center=center_small)
                corrected_r = cv2.warpAffine(frame, M_r, (w_s, h_s))
                ground_r = corrected_r[y_cut:, :].astype(np.float32)
                
                diff_t = (ref_ground - ground_t) ** 2
                diff_r = (ref_ground - ground_r) ** 2
                
                # Edge SSD (left + right 25%)
                ssd_t_edge += np.mean(diff_t[:, :edge_w]) + np.mean(diff_t[:, -edge_w:])
                ssd_r_edge += np.mean(diff_r[:, :edge_w]) + np.mean(diff_r[:, -edge_w:])
                
                # Center SSD (middle 50%)
                ssd_t_center += np.mean(diff_t[:, edge_w:-edge_w])
                ssd_r_center += np.mean(diff_r[:, edge_w:-edge_w])
                
                n_samples += 1
            
            ssd_t_edge /= max(n_samples, 1)
            ssd_r_edge /= max(n_samples, 1)
            ssd_t_center /= max(n_samples, 1)
            ssd_r_center /= max(n_samples, 1)
            
            edge_imp = (1.0 - ssd_r_edge / ssd_t_edge) * 100 if ssd_t_edge > 0 else 0
            center_imp = (1.0 - ssd_r_center / ssd_t_center) * 100 if ssd_t_center > 0 else 0
            
            # Use rotation if edges improve at all AND center doesn't get significantly worse
            use_rotation = (ssd_r_edge < ssd_t_edge) and (ssd_r_center <= ssd_t_center * 1.01)
            
            rot_range_raw = np.rad2deg(a_vals.max() - a_vals.min())
            rot_range_guide = np.rad2deg(a_guide.max() - a_guide.min())
            
            print(f"[*] Rotation validation ({n_samples} samples):")
            print(f"    Edge  SSD - translate: {ssd_t_edge:.1f}, with rotation: {ssd_r_edge:.1f} ({edge_imp:+.2f}%)")
            print(f"    Center SSD - translate: {ssd_t_center:.1f}, with rotation: {ssd_r_center:.1f} ({center_imp:+.2f}%)")
            print(f"    Rotation range - raw: {rot_range_raw:.4f}deg, guide: {rot_range_guide:.4f}deg")
            
            if use_rotation:
                print(f"    -> Using translation + rotation (rotation improves alignment)")
                corrections = corr_rotate
            else:
                print(f"    -> Using translation-only (rotation does NOT improve alignment)")
                corrections = [(dtx, dty) for dtx, dty in corr_translate]
            
            # Free memory
            del gray_frames
        
        print(f"\nMotion capture complete. {smoothing_msg}")
    else:
        print("Phase 2: Motion Capture [Skipped]")

    # --- PHASE 3: Parallel Stacking ---
    print(f"Phase 3: Stacking {total_files} files using {threads or 'default'} threads...")
    stacked_image = None
    start_time = time.time()
    
    # We use a ThreadPoolExecutor for concurrent RAW processing
    # But we stack the results sequentially to avoid race conditions on the array
    video_writer = None
    video_path = None
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        # Submit all tasks
        futures = []
        for i, filename in enumerate(files):
            corr = corrections[i]
            futures.append(executor.submit(
                process_single_frame, i, filename, src_dir, export_dir, corr,
                bright=bright
            ))
            
        # If video is requested, we MUST process results in order to ensure temporal growth
        results_iterator = futures if video_mode else concurrent.futures.as_completed(futures)
        
        for i, future in enumerate(results_iterator):
            f_idx, rgb = future.result()
            
            if rgb is not None:
                if stacked_image is None:
                    stacked_image = rgb.copy().astype(np.uint8)
                    if video_mode:
                        h_f, w_f = stacked_image.shape[:2]
                        # Determine video dimensions (downscale if necessary for codec compatibility)
                        vw, vh = w_f, h_f
                        if video_width > 0 and w_f > video_width:
                            scale_v = video_width / w_f
                            vw = video_width
                            vh = int(h_f * scale_v)
                        
                        # Use mp4v codec for broad compatibility
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        
                        base_out = os.path.splitext(output_path)[0]
                        if video_mode == 'no-trail':
                            if ground_stabilize or star_stabilize:
                                video_path = f"{base_out}{SUFFIX_STABILIZED}.mp4"
                            else:
                                video_path = f"{base_out}{SUFFIX_TIMELAPSE}.mp4"
                        else:
                            video_path = f"{base_out}.mp4"
                            
                        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (vw, vh))
                        if not video_writer.isOpened():
                            print(f"\n[!] Warning: Could not open VideoWriter for {video_path}")
                        
                        # Store target size for resizing later
                        video_target_size = (vw, vh)
                else:
                    np.maximum(stacked_image, rgb, out=stacked_image)
                
                if video_writer and video_writer.isOpened():
                    # For 'trail', we write the current stack state. For 'no-trail', just the current frame.
                    source_frame = stacked_image if video_mode == 'trail' else rgb
                    
                    frame_v = source_frame
                    if frame_v.shape[1] != video_target_size[0]:
                        frame_v = cv2.resize(frame_v, video_target_size)
                    
                    # Ensure uint8 for VideoWriter
                    if frame_v.dtype != np.uint8:
                        frame_v = np.clip(frame_v, 0, 255).astype(np.uint8)
                        
                    # OpenCV expects BGR
                    video_writer.write(cv2.cvtColor(frame_v, cv2.COLOR_RGB2BGR))
                
                if save_steps:
                    Image.fromarray(stacked_image).save(output_path, "JPEG", quality=90)
            
            progress = f"[{i+1}/{total_files}] Combined: {files[f_idx]}"
            elapsed = time.time() - start_time
            avg = elapsed / (i+1)
            rem = avg * (total_files - (i+1))
            print(f"\r{progress[:80]:<80} ({format_time(elapsed)} elapsed, ~{format_time(rem)} left)", end="", flush=True)

    if video_writer:
        video_writer.release()
        print(f"\n[*] Video saved to: {video_path}")

    if stacked_image is not None:
        print(f"\nSaving resulting stack to {output_path}...")
        img = Image.fromarray(stacked_image)
        img.save(output_path, "JPEG", quality=95, optimize=True, subsampling=0)
        
        # Copy EXIF metadata (dates, camera model, lens, GPS, etc.) from the last input file
        last_file = os.path.join(src_dir, files[-1])
        script_dir = os.path.dirname(os.path.abspath(__file__))
        exiftool_path = os.path.join(script_dir, EXIFTOOL_PATH)
        if not os.path.exists(exiftool_path):
            exiftool_path = "exiftool"
        try:
            subprocess.run(
                [exiftool_path, "-TagsFromFile", last_file,
                 "-All:All", "--IFD1:All",
                 "-overwrite_original", output_path],
                capture_output=True, text=True
            )
        except Exception:
            pass  # Non-critical, don't fail the stack
        
        end_time = time.time()
        print(f"Success! Total time: {format_time(end_time - start_time)}")
    else:
        print("\nFailed to create stack.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stack ARW files into a star trail image.")
    parser.add_argument("src_dir", help="Directory containing ARW files")
    parser.add_argument("-o", "--output", help="Output JPG filename (default: star_trail.jpg or relative to data/)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files to process")
    parser.add_argument("--save-jpg", action="store_true", help="For ARW inputs, save converted frames to a separate folder")
    parser.add_argument("--ground-stabilize", action="store_true", help="Remove high-frequency jitter using smoothed ground tracking")
    parser.add_argument("--star-stabilize", action="store_true", help="Rotate/Lock geometry around the brightest celestial point")
    parser.add_argument("--save-steps", action="store_true", help="Continuously update the output file after every frame")
    parser.add_argument("--bright", type=float, default=1.0, help="Brightness boost for the final stack (default: 1.0)")
    parser.add_argument("--threads", type=int, default=os.cpu_count(), help="Number of threads for parallel RAW processing")
    parser.add_argument("--overwrite", action="store_true", help="Automatically overwrite existing output file")
    parser.add_argument("--skip", action="store_true", help="Automatically skip if output file exists")
    parser.add_argument("--smooth", type=int, default=None, help=f"Radius for Gaussian smoothing of the motion curve (default: {int(DEFAULT_SMOOTH_FACTOR * 100)}%% of seq length)")
    parser.add_argument("--brightness-limit", default=str(DEFAULT_BRIGHTNESS_LIMIT), help="Ignore frames with mean brightness above this threshold (0.0 to 1.0) or 'auto'")
    parser.add_argument("--max-rotation", type=float, default=DEFAULT_MAX_ROTATION, help=f"Max per-frame rotation in degrees for ECC stabilization. 0 = disable rotation (default: {DEFAULT_MAX_ROTATION})")
    parser.add_argument("--motion-width", type=int, default=DEFAULT_MOTION_WIDTH, help=f"Downscale resolution width to use during the motion capture phase. (default: {DEFAULT_MOTION_WIDTH})")
    parser.add_argument("--video", choices=["trail", "no-trail"], nargs="?", const="trail", help="Save the stacking process as an MP4 video. 'trail' (default) shows the growing stack, 'no-trail' shows individual stabilized frames.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the output video (default: 30)")
    parser.add_argument("--video-width", type=int, default=3840, help="Maximum width for the output video (default: 3840 for 4K). Use 0 for original resolution.")
    
    args = parser.parse_args()
    
    # Process brightness limit
    try:
        brightness_limit = float(args.brightness_limit)
    except ValueError:
        val = args.brightness_limit.lower()
        if val == 'auto':
            brightness_limit = 'auto'
        elif val == 'none':
            brightness_limit = 1.0
        else:
            print(f"Error: Invalid brightness limit: {args.brightness_limit}")
            sys.exit(1)
    
    output_path = args.output
    if not output_path:
        output_path = get_default_output(
            args.src_dir, 
            is_star_locked=args.star_stabilize,
            is_ground_locked=args.ground_stabilize,
            smooth_radius=args.smooth
        )
    elif os.path.isdir(output_path):
        # If output is a directory, use the default name inside it
        default_full = get_default_output(
            args.src_dir, 
            is_star_locked=args.star_stabilize,
            is_ground_locked=args.ground_stabilize,
            smooth_radius=args.smooth
        )
        default_filename = os.path.basename(default_full)
        output_path = os.path.join(output_path, default_filename)
    
    stack_star_trails(args.src_dir, output_path, args.limit, 
                      overwrite=args.overwrite, skip=args.skip, 
                      save_jpg=args.save_jpg,
                      save_steps=args.save_steps,
                      bright=args.bright, threads=args.threads,
                      star_stabilize=args.star_stabilize, ground_stabilize=args.ground_stabilize,
                      smooth_radius=args.smooth, brightness_limit=brightness_limit,
                      max_rotation=args.max_rotation, motion_width=args.motion_width,
                      video_mode=args.video, fps=args.fps, video_width=args.video_width)
