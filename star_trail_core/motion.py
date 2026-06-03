import cv2
import numpy as np

from .constants import DEFAULT_MAX_ROTATION

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
