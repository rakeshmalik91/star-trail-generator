# Star Trail Stacker

`star_trail.py` - A high-performance Python utility for stacking long-exposure sequences (Sony ARW or standard images) into stunning star trail images using the 'Lighten' blend mode.

## Features

- **Ground Stabilization**: Anchors the landscape to remove wind jitter. Includes **Rotational Stabilization** (experimental) to handle tilted horizons.
- **Star Localization (experimental)**: Locks onto a celestial anchor to create perfect circular arcs and remove high-frequency shake.
- **Time-Aware Smoothing**: Uses EXIF timestamps to intelligently smooth motion curves across varying intervals.
- **Real-time Monitoring**: Watch your trails grow with the `--save-steps` flag.
- **Automatic Metadata**: Copies camera and lens metadata from the source sequence to the final stack.

## Installation

### Dependencies
```bash
pip install -r requirements.txt
```

### Specialized Tools
- **ExifTool**: Required for metadata copy. It should be located in `../lib/exif-tools/exiftool.exe`.

## Usage

### Basic Stacking
```bash
python star_trail.py "path/to/frames"
```

### Removing Wind Jitter (Ground Stabilization)
```bash
python star_trail.py "path/to/frames" --ground-stabilize --save-steps --overwrite
```

### Results Comparison
| Standard Stacking | Ground Stabilized (Locked) |
| :---: | :---: |
| <img src="data/2025-09-27%20-%20Hali%20Country%20TL2_star_trail_resized.jpg" width="500"> | <img src="data/2025-09-27%20-%20Hali%20Country%20TL2_star_trail_ground_locked_sm300_resized.jpg" width="500"> |



### Circular Arcs / Star Tracking (experimental)
```bash
python star_trail.py "path/to/frames" --star-stabilize --limit 100
```
> [!NOTE]
> This mode is currently experimental and requires clear visibility of the celestial pivot point.

### Advanced Options

#### General Options
- `-o`, `--output <path>`: Specific output JPG filename or directory. If omitted or set to a directory, defaults to a file named after the source folder with appropriate suffixes.
- `--limit <int>`: Limit the number of frames processed from the input sequence.
- `--overwrite`: Automatically overwrite existing output files without prompting.
- `--skip`: Automatically skip processing if the output file already exists.
- `--threads <int>`: Number of concurrent RAW processing threads (defaults to CPU count).
- `--bright <float>`: Apply a brightness boost factor to the frames during processing (default: 1.0).
- `--save-steps`: Continuously update and save the output JPG file after processing every frame, allowing real-time progress monitoring.
- `--save-jpg`: Save converted JPEG frames from RAW ARW inputs into a separate folder (suffix `_jpg`).

#### Stabilization & Filtering
- `--ground-stabilize`: Remove high-frequency wind jitter by anchoring the landscape using bottom-masked phase correlation and motion smoothing.
- `--star-stabilize`: Align frames dynamically by tracking the brightest celestial anchor to ensure perfect circular star trails.
- `--smooth <int>`: Set a specific radius (in frames) for Gaussian motion curve smoothing (defaults to 100% of the sequence length).
- `--brightness-limit <value>`: Exclude overexposed frames (e.g. headlights, clouds) using a threshold (0.0 to 1.0) or `auto` (uses standard deviation), or `none`/`1.0` to disable.
- `--max-rotation <float>`: Maximum allowed per-frame rotation in degrees for ECC stabilization (default: 5.0). Use `0` to disable rotation tracking.
- `--motion-width <int>`: Downscaled image width used during the motion detection phase to speed up analysis (default: 1500).

#### Video Generation
- `--video [trail|no-trail]`: Generate an MP4 video of the sequence. `trail` (default if flag is set) renders the progressive stack growing, while `no-trail` outputs the individual stabilized frames.
- `--fps <int>`: Frame rate for the output video (default: 30).
- `--video-width <int>`: Maximum width of the output video in pixels (default: 3840 for 4K). Set to `0` to preserve the original sequence resolution.

## Data Folder
The `data/` folder typically contains sample sequences or output files when following default naming conventions.
