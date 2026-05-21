import numpy as np
import cv2
from scipy.stats import entropy as scipy_entropy
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────────────────
# PARAMETERS FOR FEATURE DETECTION AND OPTICAL FLOW TRACKING
# ────────────────────────────────────────────────────────────────────────────
# Parameters for good features to track detector
feature_params = dict(maxCorners=10000,        # Maximum number of features to detect
                      qualityLevel=0.01,        # Quality threshold for feature detection
                      minDistance=5,            # Minimum distance between features
                      blockSize=7)              # Window size for feature detection

# Parameters for Lucas-Kanade optical flow algorithm
lk_params = dict(winSize=(15, 15),            # Window size for optical flow calculation
                 maxLevel=8,                   # Number of pyramid levels
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.003))  # Convergence criteria

# ────────────────────────────────────────────────────────────────────────────
# INITIALIZE VIDEO CAPTURE
# ────────────────────────────────────────────────────────────────────────────
# Load video from file
cap = cv2.VideoCapture("../Projet2_Videos/Extrait5-Matrix-Helicopter_Scene(280p).m4v")
# cap = cv2.VideoCapture(0)  # Alternative: Use webcam (uncomment to use)

# Read first frame and convert to grayscale
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
h, w = old_gray.shape  # Get image dimensions (height, width)

# Detect initial features in the first frame
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# List to store entropy values for each frame
entropy_list = []
frame_index = 1  # Frame counter

# ────────────────────────────────────────────────────────────────────────────
# DENSE IMAGE WARPING FUNCTION
# ────────────────────────────────────────────────────────────────────────────
def warp_image(prev_gray, flow):
    """
    Reconstructs the predicted image I_hat_t from the previous image I_{t-1}
    using the dense optical flow field.
    
    Parameters:
    -----------
    prev_gray : ndarray
        Previous grayscale frame (H, W)
    flow : ndarray
        Dense optical flow field (H, W, 2) containing velocity vectors (Vx, Vy)
    
    Returns:
    --------
    I_hat : ndarray
        Warped/predicted image reconstructed from flow
    """
    
    # Create coordinate grids for pixel mapping
    # For each pixel (x, y), calculate its source position (x + Vx, y + Vy)
    map_x = (np.arange(w, dtype=np.float32)[None, :] + flow[..., 0])
    map_y = (np.arange(h, dtype=np.float32)[:, None] + flow[..., 1])
    
    # Use cv2.remap to perform image warping based on calculated coordinates
    I_hat = cv2.remap(prev_gray,                          # Source image
                      map_x, map_y,                        # Coordinate maps
                      interpolation=cv2.INTER_LINEAR,     # Linear interpolation
                      borderMode=cv2.BORDER_REPLICATE)    # Handle border pixels
    return I_hat

# ────────────────────────────────────────────────────────────────────────────
# IMAGE ENTROPY CALCULATION FUNCTION
# ────────────────────────────────────────────────────────────────────────────
def image_entropy(img):
    """
    Calculate Shannon entropy of a grayscale image based on its intensity histogram.
    
    Parameters:
    -----------
    img : ndarray
        Input grayscale image (0-255)
    
    Returns:
    --------
    float
        Shannon entropy in bits
    """
    # Calculate histogram of grayscale values (256 bins)
    hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256))
    
    # Normalize histogram to get probability distribution
    hist = hist / hist.sum()
    
    # Remove zero probabilities to avoid log(0)
    hist = hist[hist > 0]
    
    # Calculate Shannon entropy: H = -sum(p * log2(p))
    return -np.sum(hist * np.log2(hist))

# ────────────────────────────────────────────────────────────────────────────
# MAIN PROCESSING LOOP
# ────────────────────────────────────────────────────────────────────────────
while ret:
    # Read next frame
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if no more frames
    
    # Convert current frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_index += 1

    # ────────────────────────────────────────────────────────────────────────
    # STEP 1: COMPUTE DENSE OPTICAL FLOW (Farneback method)
    # ────────────────────────────────────────────────────────────────────────
    # Farneback algorithm computes optical flow for all pixels at once
    # (as opposed to Lucas-Kanade which only tracks sparse feature points)
    flow = cv2.calcOpticalFlowFarneback(
        old_gray, frame_gray,           # Previous and current frames
        None,                           # No initial flow guess
        pyr_scale=0.5,                  # Scale factor for image pyramid
        levels=lk_params['maxLevel'],   # Number of pyramid levels
        winsize=15,                     # Window size for averaging
        iterations=3,                   # Number of iterations
        poly_n=5,                       # Number of polynomial expansion terms
        poly_sigma=1.2,                 # Standard deviation of Gaussian
        flags=0                         # No flags set
    )

    # ────────────────────────────────────────────────────────────────────────
    # STEP 2: COMPUTE PREDICTED IMAGE AND RESIDUAL
    # ────────────────────────────────────────────────────────────────────────
    # Warp previous frame according to optical flow to predict current frame
    I_hat = warp_image(old_gray, flow)
    
    # Calculate residual (difference between actual and predicted frame)
    # Using int16 to handle negative values from subtraction
    R = frame_gray.astype(np.int16) - I_hat.astype(np.int16)

    # ────────────────────────────────────────────────────────────────────────
    # STEP 3: MEASURE ENTROPY OF RESIDUAL
    # ────────────────────────────────────────────────────────────────────────
    # Take absolute value of residual to get magnitude
    R_abs = np.abs(R).astype(np.uint8)
    
    # Calculate entropy of the residual image
    H = image_entropy(R_abs)
    
    # Store entropy value for later analysis
    entropy_list.append(H)

    # ────────────────────────────────────────────────────────────────────────
    # STEP 4: DISPLAY RESULTS
    # ────────────────────────────────────────────────────────────────────────
    # Convert residual for display (centered at 128 for better visualization)
    R_display = cv2.convertScaleAbs(R, alpha=1, beta=128)
    
    # Show three windows: original frame, predicted image, and residual
    cv2.imshow('Current Frame', frame_gray)
    cv2.imshow('Predicted Image I_hat', I_hat)
    cv2.imshow('Residual R_t (centered at 128)', R_display)

    # Handle keyboard input (30ms timeout)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:  # ESC key
        break  # Exit program
    elif k == ord('s'):  # 's' key
        # Save residual image with frame number
        cv2.imwrite(f'residual_{frame_index:04d}.png', R_display)

    # ────────────────────────────────────────────────────────────────────────
    # STEP 5: UPDATE FOR NEXT ITERATION
    # ────────────────────────────────────────────────────────────────────────
    # Update previous frame for next iteration
    old_gray = frame_gray.copy()
    
    # Re-detect features in current frame for next iteration
    if p0 is not None:
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)

# Close all OpenCV windows
cv2.destroyAllWindows()
# Release video capture object
cap.release()

# ────────────────────────────────────────────────────────────────────────────
# SCENE CUT DETECTION AND ANALYSIS
# ────────────────────────────────────────────────────────────────────────────
# Convert entropy list to numpy array for analysis
entropy_arr = np.array(entropy_list)

# ────────────────────────────────────────────────────────────────────────────
# VISUALIZATION OF ENTROPY OVER TIME
# ────────────────────────────────────────────────────────────────────────────
# Create a figure with one subplot
plt.figure(figsize=(14, 4))

# Plot entropy values for each frame
plt.plot(entropy_arr,                           # Entropy values
         color='steelblue',                     # Line color
         linewidth=0.8,                         # Line thickness
         label='Residual Entropy $H(R_t)$')     # Legend label

# Labels and title
plt.xlabel('Frame Index')
plt.ylabel('Entropy (bits)')
plt.title('Residual Image Entropy — Scene Cut Detection')

# Add legend and adjust layout
plt.legend()
plt.tight_layout()

# Save and display the plot
plt.savefig('entropy_cuts.png', dpi=150)
plt.show()
