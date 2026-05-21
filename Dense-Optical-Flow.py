import math
import cv2
import numpy as np

# ────────────────────────────────────────────────────────────────────────────
# INITIALIZE VIDEO CAPTURE
# ────────────────────────────────────────────────────────────────────────────
# Load video from file for dense optical flow analysis
cap = cv2.VideoCapture("../Projet2_Videos/Extrait5-Matrix-Helicopter_Scene(280p).m4v")
# Alternative: Uncomment line below to use webcam instead of video file
# cap = cv2.VideoCapture(0)

# ────────────────────────────────────────────────────────────────────────────
# READ INITIAL FRAMES FOR OPTICAL FLOW CALCULATION
# ────────────────────────────────────────────────────────────────────────────
# Read the first frame from the video
ret, frame1 = cap.read()

# Convert first frame to grayscale (required for optical flow calculation)
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# ────────────────────────────────────────────────────────────────────────────
# INITIALIZE HSV IMAGE FOR FLOW VISUALIZATION
# ────────────────────────────────────────────────────────────────────────────
# Create HSV (Hue, Saturation, Value) image with same dimensions as frame
# HSV representation: Hue (angle) = flow direction, Value (brightness) = flow magnitude
hsv = np.zeros_like(frame1)

# Set saturation to maximum (255) for vivid color representation of motion
hsv[:, :, 1] = 255

# Frame counter to track position in video
index = 1

# Read the second frame to begin optical flow calculation
ret, frame2 = cap.read()

# Convert second frame to grayscale
next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# ────────────────────────────────────────────────────────────────────────────
# MAIN PROCESSING LOOP - PROCESS VIDEO FRAME BY FRAME
# ────────────────────────────────────────────────────────────────────────────
while(ret):
    # Increment frame counter
    index += 1
    
    # ────────────────────────────────────────────────────────────────────────
    # STEP 1: CALCULATE DENSE OPTICAL FLOW (FARNEBACK METHOD)
    # ────────────────────────────────────────────────────────────────────────
    # Farneback calculates optical flow for every pixel in the image
    # This provides a complete velocity field for the entire frame
    flow = cv2.calcOpticalFlowFarneback(
        prvs, next,                    # Previous and current frames
        None,                          # No initial flow estimate
        pyr_scale=0.5,                 # Scale factor for image pyramid
        levels=3,                      # Number of pyramid levels
        winsize=15,                    # Window size for averaging
        iterations=3,                  # Number of iterations at each pyramid level
        poly_n=7,                      # Polynomial expansion size
        poly_sigma=1.5,                # Gaussian standard deviation
        flags=0                        # No special flags
    )

    # ────────────────────────────────────────────────────────────────────────
    # STEP 2: CONVERT FLOW TO MAGNITUDE AND ANGLE
    # ────────────────────────────────────────────────────────────────────────
    # Convert Cartesian velocity components (Vx, Vy) to polar coordinates (magnitude, angle)
    # This is necessary for HSV visualization where angle represents direction
    # and magnitude represents speed
    mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
    
    # ────────────────────────────────────────────────────────────────────────
    # STEP 3: POPULATE HSV IMAGE FOR VISUALIZATION
    # ────────────────────────────────────────────────────────────────────────
    # Hue channel: flow direction (angle in degrees, 0-180 for OpenCV HSV)
    hsv[:, :, 0] = (ang * 180) / (2 * np.pi)  # Convert radians to degrees
    
    # Value channel: flow magnitude (0-255), normalized by maximum magnitude
    hsv[:, :, 2] = (mag * 255) / np.amax(mag)  # Normalize magnitude to 0-255 range

    # ────────────────────────────────────────────────────────────────────────
    # STEP 4: EXTRACT VELOCITY COMPONENTS FOR HISTOGRAM
    # ────────────────────────────────────────────────────────────────────────
    # Extract Cartesian velocity components from optical flow field
    vx = flow[:, :, 0].astype(np.float32)  # Horizontal velocity component
    vy = flow[:, :, 1].astype(np.float32)  # Vertical velocity component

    # ────────────────────────────────────────────────────────────────────────
    # STEP 5: CREATE 2D VELOCITY HISTOGRAM
    # ────────────────────────────────────────────────────────────────────────
    # Build a 2D histogram representing the joint distribution of (Vx, Vy) velocities
    # This shows how often different velocity combinations occur in the frame
    hist = cv2.calcHist(
        images=[vx, vy],               # Input velocity channels
        channels=[0, 1],               # Use both vx and vy channels
        mask=None,                     # No mask
        histSize=[256, 256],           # 256x256 bins for velocity space
        ranges=[-10, 30, -15, 15]      # Vx range: -10 to 30 pixels, Vy range: -15 to 15 pixels
    )

    # ────────────────────────────────────────────────────────────────────────
    # STEP 6: CALCULATE JOINT PROBABILITY DISTRIBUTION
    # ────────────────────────────────────────────────────────────────────────
    # Normalize histogram to get probability distribution
    # This represents the likelihood of each (Vx, Vy) velocity pair occurring
    hist_prob = hist / hist.sum()

    # ────────────────────────────────────────────────────────────────────────
    # STEP 7: VISUALIZE VELOCITY HISTOGRAM AS HEATMAP
    # ────────────────────────────────────────────────────────────────────────
    # Apply logarithmic scaling to histogram for better contrast visualization
    # Highlights low-probability regions that would otherwise be invisible
    hist_log = np.log1p(hist)  # log(1 + hist) to avoid log(0)
    
    # Normalize logarithmic histogram to 0-255 range for display
    hist_log_normalized = (hist_log * 255.0 / np.amax(hist_log)).astype(np.uint8)
    
    # Apply JET colormap to convert grayscale to color (blue=low, red=high)
    entropy_colored = cv2.applyColorMap(hist_log_normalized, cv2.COLORMAP_JET)
    
    # Resize histogram visualization to 512x512 pixels for better viewing
    entropy_resized = cv2.resize(entropy_colored, (512, 512), interpolation=cv2.INTER_NEAREST)
    
    # ────────────────────────────────────────────────────────────────────────
    # STEP 8: CALCULATE SHANNON ENTROPY OF VELOCITY FIELD
    # ────────────────────────────────────────────────────────────────────────
    # Entropy measures the disorder/uncertainty in motion patterns
    # High entropy = diverse motion directions, Low entropy = uniform motion
    hist_prob_flat = hist_prob[hist_prob > 0]  # Remove zero probabilities (avoid log(0))
    
    # Calculate Shannon entropy: H = -sum(p * log2(p)) in bits
    total_entropy = -np.sum(hist_prob_flat * np.log2(hist_prob_flat))
    
    # ────────────────────────────────────────────────────────────────────────
    # STEP 9: DRAW CROSSHAIR AT ORIGIN (VX=0, VY=0)
    # ────────────────────────────────────────────────────────────────────────
    # The crosshair marks the point where there is zero motion (no flow)
    # This helps identify regions of motion vs static regions
    
    # Calculate x-coordinate of zero velocity in the histogram display
    # Vx range: -10 to 30, so 0 is at position (0 - (-10)) / (30 - (-10)) = 10/40 = 0.25 * 512 ≈ 128 px
    origin_x = int((0 - (-10)) / (30 - (-10)) * 512)
    
    # Calculate y-coordinate of zero velocity in the histogram display
    # Vy range: -15 to 15, so 0 is at position (0 - (-15)) / (15 - (-15)) = 15/30 = 0.5 * 512 = 256 px
    origin_y = int((0 - (-15)) / (15 - (-15)) * 512)
    
    # Draw vertical line at Vx=0
    cv2.line(entropy_resized, (origin_x, 0), (origin_x, 512), (100, 100, 100), 1)
    
    # Draw horizontal line at Vy=0
    cv2.line(entropy_resized, (0, origin_y), (512, origin_y), (100, 100, 100), 1)

    # Display entropy visualization window
    cv2.imshow('Velocity Field Entropy - Farneback', entropy_resized)

    # ────────────────────────────────────────────────────────────────────────
    # STEP 10: DISPLAY MAIN OPTICAL FLOW VISUALIZATION
    # ────────────────────────────────────────────────────────────────────────
    # Convert HSV image back to BGR color space for display
    # HSV format where: Hue=direction, Saturation=255 (vivid), Value=magnitude
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Stack current frame and optical flow visualization vertically for comparison
    result = np.vstack((frame2, bgr))
    
    # Display stacked result (top: original frame, bottom: optical flow)
    cv2.imshow('Frame and Velocity Field - Farneback', result)

    # ────────────────────────────────────────────────────────────────────────
    # STEP 11: HANDLE KEYBOARD INPUT
    # ────────────────────────────────────────────────────────────────────────
    k = cv2.waitKey(15) & 0xff
    
    if k == 27:  # ESC key - exit program
        break
    elif k == ord('s'):  # 's' key - save current frame and visualizations
        cv2.imwrite('Frame_%04d.png' % index, frame2)  # Save current frame
        cv2.imwrite('OF_hsv_%04d.png' % index, bgr)    # Save optical flow in HSV representation
        cv2.imwrite('OF_entropy_%04d.png' % index, entropy_resized)  # Save entropy histogram

    # ────────────────────────────────────────────────────────────────────────
    # STEP 12: UPDATE FOR NEXT ITERATION
    # ────────────────────────────────────────────────────────────────────────
    # Shift frames for next iteration: current becomes previous
    prvs = next
    
    # Read next frame from video
    ret, frame2 = cap.read()
    
    # Convert to grayscale if frame was successfully read
    if ret:
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# ────────────────────────────────────────────────────────────────────────────
# CLEANUP - RELEASE RESOURCES
# ────────────────────────────────────────────────────────────────────────────
# Release the video capture object and free system resources
cap.release()

# Close all OpenCV display windows
cv2.destroyAllWindows()
