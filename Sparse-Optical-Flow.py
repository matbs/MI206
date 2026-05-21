import numpy as np
import cv2

# ────────────────────────────────────────────────────────────────────────────
# INITIALIZE VIDEO CAPTURE
# ────────────────────────────────────────────────────────────────────────────
# Load video from file
cap = cv2.VideoCapture('../Projet2_Videos/Extrait2-ManWithAMovieCamera.m4v')
# Alternative: Uncomment line below to use webcam instead of video file
# cap = cv2.VideoCapture(0)

# ────────────────────────────────────────────────────────────────────────────
# PARAMETERS FOR FEATURE DETECTION AND OPTICAL FLOW TRACKING
# ────────────────────────────────────────────────────────────────────────────
# Parameters for 'Good Features to Track' detector (Shi-Tomasi corners)
feature_params = dict( maxCorners = 10000,        # Maximum number of features to detect
                       qualityLevel = 0.01,        # Quality threshold for feature detection
                       minDistance = 5,            # Minimum distance between features (in pixels)
                       blockSize = 7 )             # Window size for corner detection

# Parameters for Lucas-Kanade Pyramidal optical flow algorithm
lk_params = dict( winSize  = (15,15),              # Window size for flow calculation
                  maxLevel = 8,                     # Number of pyramid levels (0=no pyramid)
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.003))  # Convergence criteria

# ────────────────────────────────────────────────────────────────────────────
# READ INITIAL FRAME AND DETECT FEATURES
# ────────────────────────────────────────────────────────────────────────────
# Read the first frame from the video
ret, old_frame = cap.read()

# Convert first frame to grayscale (required for optical flow calculation)
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Detect good features to track in the first frame
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Frame counter to track position in video
index = 1

# ────────────────────────────────────────────────────────────────────────────
# MAIN PROCESSING LOOP - PROCESS VIDEO FRAME BY FRAME
# ────────────────────────────────────────────────────────────────────────────
while(ret):
    # Read next frame from video
    ret, frame = cap.read()
    
    # Convert current frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Increment frame counter
    index += 1

    # ────────────────────────────────────────────────────────────────────────
    # STEP 1: INITIALIZE FEATURE TRACKING VARIABLES
    # ────────────────────────────────────────────────────────────────────────
    # Initialize arrays to store tracked feature points
    good_new = np.array([])  # Tracked features in current frame
    good_old = np.array([])  # Tracked features in previous frame
    
    # ────────────────────────────────────────────────────────────────────────
    # STEP 2: CALCULATE SPARSE OPTICAL FLOW (LUCAS-KANADE)
    # ────────────────────────────────────────────────────────────────────────
    # Only calculate flow if features were successfully detected in previous frame
    if p0 is not None:
        # Calculate optical flow using Lucas-Kanade pyramidal method
        # Returns: new points (p1), status array (st), error values (err)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select only valid tracked points (where status == 1)
        good_new = p1[st==1]  # Feature positions in current frame
        good_old = p0[st==1]  # Feature positions in previous frame
    
    # ────────────────────────────────────────────────────────────────────────
    # STEP 3: CALCULATE ENTROPY FROM SPARSE FLOW VECTORS
    # ────────────────────────────────────────────────────────────────────────
    # Only calculate entropy if features were successfully tracked
    if len(good_new) > 0:
        # Compute velocity vectors (displacement) for each tracked feature
        # vx = horizontal displacement (change in x-coordinate)
        # vy = vertical displacement (change in y-coordinate)
        vx = (good_new[:, 0] - good_old[:, 0]).astype(np.float32)
        vy = (good_new[:, 1] - good_old[:, 1]).astype(np.float32)
        
        # ────────────────────────────────────────────────────────────────────
        # Create 2D histogram of velocity components (Vx, Vy)
        # This represents the probability distribution of motion vectors
        hist = cv2.calcHist(
            images=[vx, vy],                  # Input velocity channels
            channels=[0, 1],                  # Use both vx and vy
            mask=None,                        # No mask
            histSize=[256, 256],              # 256x256 bins for velocity space
            ranges=[-20, 20, -20, 20]         # Velocity range: -20 to 20 pixels
        )
        
        # ────────────────────────────────────────────────────────────────────
        # Calculate joint probability distribution from histogram
        hist_prob = hist / hist.sum()  # Normalize to get probabilities
        
        # ────────────────────────────────────────────────────────────────────
        # Calculate Shannon entropy of the velocity field
        # Entropy measures the disorder/uncertainty in motion patterns
        hist_prob_flat = hist_prob[hist_prob > 0]  # Remove zero probabilities (avoid log(0))
        total_entropy = -np.sum(hist_prob_flat * np.log2(hist_prob_flat))  # Shannon entropy in bits
        
        # ────────────────────────────────────────────────────────────────────
        # Visualize entropy as a 2D heatmap
        # Apply logarithmic scaling to enhance visibility of low-probability regions
        hist_log = np.log1p(hist)  # log(1 + hist) to avoid log(0)
        hist_log_normalized = (hist_log * 255.0 / np.amax(hist_log)).astype(np.uint8)
        
        # Apply colormap to histogram for better visualization
        entropy_colored = cv2.applyColorMap(hist_log_normalized, cv2.COLORMAP_JET)
        
        # Resize histogram visualization to 512x512 for better viewing
        entropy_resized = cv2.resize(entropy_colored, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        # ────────────────────────────────────────────────────────────────────
        # Draw crosshair at origin (Vx=0, Vy=0) in the velocity histogram
        # This helps identify zero motion regions
        origin_x = int((0 - (-20)) / (20 - (-20)) * 512)  # Calculate x position (~256)
        origin_y = int((0 - (-20)) / (20 - (-20)) * 512)  # Calculate y position (~256)
        cv2.line(entropy_resized, (origin_x, 0), (origin_x, 512), (100, 100, 100), 1)  # Vertical line
        cv2.line(entropy_resized, (0, origin_y), (512, origin_y), (100, 100, 100), 1)  # Horizontal line
        
        # Display entropy visualization
        cv2.imshow('Velocity Field Entropy - Lucas-Kanade', entropy_resized)
    
    # ────────────────────────────────────────────────────────────────────────
    # STEP 4: DRAW OPTICAL FLOW VECTORS ON THE FRAME
    # ────────────────────────────────────────────────────────────────────────
    # Create a mask image for drawing flow vectors
    mask = np.zeros_like(old_frame)

    # Draw motion vectors for each tracked feature
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        # Extract coordinates of tracked point in current frame
        a,b = new.ravel().astype(int)
        
        # Extract coordinates of tracked point in previous frame
        c,d = old.ravel().astype(int)
        
        # Draw line from old position to new position (motion vector)
        mask = cv2.line(mask, (a,b),(c,d),(255,255,0),2)
        
        # Draw circle at old position to mark the feature point
        frame = cv2.circle(frame,(c,d),3,(255,255,0),-1)
    
    # Combine frame and motion vectors
    img = cv2.add(frame, mask)

    # Display the result
    cv2.imshow('Sparse Optical Flow - Lucas-Kanade Pyramidal', img)
    
    # ────────────────────────────────────────────────────────────────────────
    # STEP 5: HANDLE KEYBOARD INPUT
    # ────────────────────────────────────────────────────────────────────────
    k = cv2.waitKey(30) & 0xff
    
    if k == 27:  # ESC key - exit program
        break
    elif k == ord('s'):  # 's' key - save current frame and results
        cv2.imwrite('Frame_%04d.png' % index, frame)  # Save raw frame
        cv2.imwrite('OF_PyrLk_%04d.png' % index, img)  # Save frame with flow vectors
        if len(good_new) > 0:
            cv2.imwrite('OF_entropy_sparse_%04d.png' % index, entropy_resized)  # Save entropy visualization

    # ────────────────────────────────────────────────────────────────────────
    # STEP 6: UPDATE FOR NEXT ITERATION
    # ────────────────────────────────────────────────────────────────────────
    # Detect new features in current frame for next iteration
    p0 = cv2.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)
    # Update previous frame for next iteration
    old_gray = frame_gray.copy()

# ────────────────────────────────────────────────────────────────────────────
# CLEANUP - RELEASE RESOURCES
# ────────────────────────────────────────────────────────────────────────────
# Close all OpenCV display windows
cv2.destroyAllWindows()

# Release the video capture object and free system resources
cap.release()
