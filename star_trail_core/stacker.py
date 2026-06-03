import os
import time
import subprocess
import concurrent.futures
import numpy as np
import rawpy
import cv2
from PIL import Image, ImageOps

from .constants import (
    EXIFTOOL_PATH,
    SUFFIX_STAR_LOCKED,
    SUFFIX_GROUND_LOCKED,
    SUFFIX_DEFAULT,
    SUFFIX_SMOOTH,
    SUFFIX_JPG_DIR,
    DEFAULT_FOLDER_NAME,
    SUFFIX_STABILIZED,
    SUFFIX_TIMELAPSE,
    MIN_TIMESTAMP_DIFF,
    DEFAULT_BRIGHTNESS_LIMIT,
    AUTO_BRIGHTNESS_FACTOR,
    DEFAULT_SMOOTH_MIN,
    DEFAULT_SMOOTH_FACTOR,
    DEFAULT_MAX_ROTATION,
    DEFAULT_MOTION_WIDTH,
)

from .utils import (
    format_time,
    smooth_curve,
    get_exif_timestamps,
    smooth_curve_time_aware,
    get_brightness,
    get_default_output,
)

from .motion import (
    get_brightest_star,
    track_star_patch,
    compose_affine,
    estimate_ground_motion,
)

def process_single_frame(idx, filename, src_dir, export_dir, correction, bright=1.0):
    """
    Worker function. Applies sub-pixel translation or perspective correction.
    """
    path = os.path.join(src_dir, filename)
    try:
        rgb = None
        if filename.lower().endswith('.arw'):
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
                if len(clean_vals) > 0:
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
                        vw, vh = w_f, h_f
                        if video_width > 0 and w_f > video_width:
                            scale_v = video_width / w_f
                            vw = video_width
                            vh = int(h_f * scale_v)
                        
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
                        
                        video_target_size = (vw, vh)
                else:
                    np.maximum(stacked_image, rgb, out=stacked_image)
                
                if video_writer and video_writer.isOpened():
                    source_frame = stacked_image if video_mode == 'trail' else rgb
                    
                    frame_v = source_frame
                    if frame_v.shape[1] != video_target_size[0]:
                        frame_v = cv2.resize(frame_v, video_target_size)
                    
                    if frame_v.dtype != np.uint8:
                        frame_v = np.clip(frame_v, 0, 255).astype(np.uint8)
                        
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
        
        # Copy EXIF metadata from the last input file
        last_file = os.path.join(src_dir, files[-1])
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
