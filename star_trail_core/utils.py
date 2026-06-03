import os
import subprocess
import json
from datetime import datetime
import numpy as np
import rawpy
from PIL import Image, ImageOps

from .constants import (
    EXIFTOOL_PATH,
    SUFFIX_STAR_LOCKED,
    SUFFIX_GROUND_LOCKED,
    SUFFIX_DEFAULT,
    SUFFIX_SMOOTH,
    DEFAULT_FOLDER_NAME,
    MIN_TIMESTAMP_DIFF,
    AUTO_BRIGHTNESS_FACTOR,
    DEFAULT_SMOOTH_MIN,
)

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
    # Since this module is inside star_trail_core, we need to locate EXIFTOOL_PATH relative to the workspace root.
    # The original script does:
    #   script_dir = os.path.dirname(os.path.abspath(__file__))
    #   exiftool_path = os.path.join(script_dir, EXIFTOOL_PATH)
    # But since __file__ is in star_trail_core, script_dir here is star_trail_core.
    # The workspace root is one level up.
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
