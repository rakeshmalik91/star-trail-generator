import os

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
