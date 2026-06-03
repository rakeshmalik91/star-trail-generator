# Star Trail Stacker CLI
# ------------------
# Stacks multiple Sony ARW or standard image files into a single 'Lighten' blend stack.
# Optimized for high-resolution sequences with professional-grade stabilization.
#
# Usage:
#     python star_trail.py "path/to/frames" --ground-stabilize --bright 1.2
#     python star_trail.py "path/to/frames" --star-stabilize --limit 100
#
# Dependencies: rawpy, numpy, Pillow, opencv-python

import os
import sys
import argparse

from star_trail_core import stack_star_trails, get_default_output
from star_trail_core.constants import (
    DEFAULT_BRIGHTNESS_LIMIT,
    DEFAULT_MAX_ROTATION,
    DEFAULT_MOTION_WIDTH,
    DEFAULT_SMOOTH_FACTOR,
)

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
