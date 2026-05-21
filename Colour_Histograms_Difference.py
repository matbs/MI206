import cv2
import numpy as np
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ────────────────────────────────────────────────────────────────────────────
# Window size for moving average smoothing filter
# Larger values = more smoothing, smaller values = follows signal more closely
WINDOW_SIZE = 5

# ────────────────────────────────────────────────────────────────────────────
# SETUP MATPLOTLIB FOR REAL-TIME PLOTTING
# ────────────────────────────────────────────────────────────────────────────
# Enable interactive mode for live updating plots during execution
plt.ion()

# Create figure and axis for plotting
fig, ax = plt.subplots()

# Create three line plots for different metrics:
# 1. Raw histogram distance between consecutive frames (using L1 norm / Manhattan distance)
line_diff, = ax.plot([], [], 'r-', label='Raw Difference', alpha=0.5)

# 2. Smoothed histogram distance using moving average filter
line_smooth, = ax.plot([], [], 'b-', label='Smoothed', linewidth=2)

# 3. First derivative of smoothed signal (rate of change of histogram difference)
line_deriv, = ax.plot([], [], 'g-', label='Derivative', linewidth=2)

# Configure plot appearance
ax.set_title("Histogram Difference Over Time")
ax.set_xlabel("Frame Number")
ax.set_ylabel("Difference (Absolute Sum)")
ax.legend()

# ────────────────────────────────────────────────────────────────────────────
# INITIALIZE DATA STORAGE LISTS
# ────────────────────────────────────────────────────────────────────────────
# Lists to store plotting data for real-time visualization
x_data = []              # Frame indices
y_diff_data = []         # Raw histogram differences
y_smooth_data = []       # Smoothed histogram differences
y_derivative_data = []   # Derivatives of smoothed signal

# ────────────────────────────────────────────────────────────────────────────
# INITIALIZE VIDEO CAPTURE
# ────────────────────────────────────────────────────────────────────────────
# Load video file for color histogram analysis using L1 (Manhattan) distance
cap = cv2.VideoCapture("../Projet2_Videos/Extrait2-ManWithAMovieCamera.m4v")

# Variable to store histogram from previous frame for comparison
prev_hist = None

# Frame counter
index = 0

# ────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTION: SIGNAL SMOOTHING
# ────────────────────────────────────────────────────────────────────────────
def smooth_signal(signal, window_size):
    """
    Apply moving average smoothing filter to a signal.
    
    This reduces noise and high-frequency variations in the signal,
    making trends more visible and scene boundaries more obvious.
    
    Parameters:
    -----------
    signal : list or array
        Input signal to smooth
    window_size : int
        Size of the moving average window
    
    Returns:
    --------
    float
        Smoothed value (last point after convolution)
    """
    # Check if we have enough data points for smoothing
    if len(signal) < window_size:
        # Not enough data - return last signal value
        return signal[-1]
    
    # Create uniform kernel for moving average (equal weights)
    kernel = np.ones(window_size) / window_size
    
    # Apply convolution (moving average) and return the last smoothed value
    return np.convolve(signal, kernel, mode='valid')[-1]

# ────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTION: CALCULATE FIRST DERIVATIVE
# ────────────────────────────────────────────────────────────────────────────
def first_derivative(signal, time_interval):
    """
    Calculate the first derivative of a signal using finite differences.
    
    The derivative represents the rate of change of histogram difference.
    High derivatives indicate rapid changes in color distribution (scene cuts).
    Low derivatives indicate stable scenes with consistent colors.
    
    Parameters:
    -----------
    signal : list or array
        Input signal
    time_interval : float
        Time interval between samples (usually 1 frame)
    
    Returns:
    --------
    float
        Derivative value (rate of change)
    """
    # Need at least 2 points to calculate derivative
    if len(signal) < 2:
        return 0
    
    # Finite difference: (f(t) - f(t-1)) / dt
    return (signal[-1] - signal[-2]) / time_interval

# ────────────────────────────────────────────────────────────────────────────
# MAIN PROCESSING LOOP - ANALYZE VIDEO FRAME BY FRAME
# ────────────────────────────────────────────────────────────────────────────
while True:
    # Read next frame from video
    ret, frame = cap.read()
    
    # Check if frame was successfully read
    if not ret:
        break

    # ────────────────────────────────────────────────────────────────────────
    # STEP 1: CONVERT FRAME TO YUV COLOR SPACE
    # ────────────────────────────────────────────────────────────────────────
    # YUV separates luminance (Y) from chrominance (U, V)
    # U and V channels represent color information and are more robust to lighting changes
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    
    # ────────────────────────────────────────────────────────────────────────
    # STEP 2: CALCULATE 2D COLOR HISTOGRAM
    # ────────────────────────────────────────────────────────────────────────
    # Create a 2D histogram of U and V chrominance channels
    # This represents the color distribution in the frame
    # Histogram comparison is more robust to scene changes than pixel-level comparison
    hist = cv2.calcHist([yuv],                    # Input image (YUV frame)
                        [1, 2],                   # Channels: U (channel 1) and V (channel 2)
                        None,                     # No mask
                        [256, 256],               # Histogram bins: 256x256 for U-V space
                        [0, 256, 0, 256])         # U range: 0-256, V range: 0-256

    # ────────────────────────────────────────────────────────────────────────
    # STEP 3: COMPARE HISTOGRAMS USING L1 NORM (MANHATTAN DISTANCE)
    # ────────────────────────────────────────────────────────────────────────
    if prev_hist is not None:
        # Calculate L1 norm (Manhattan distance / Taxicab distance)
        # between current and previous histogram
        # 
        # L1 norm = sum(|hist[i] - prev_hist[i]|)
        # 
        # Properties:
        # - Simple and fast to compute
        # - Range: 0 to infinity (0 = identical histograms, larger = more different)
        # - Sensitive to overall histogram magnitude differences
        # - May overestimate differences due to normalization sensitivity
        # 
        # This method is less statistically robust than Bhattacharyya distance
        # but is computationally faster and more intuitive (direct pixel differences)
        diff = np.sum(np.abs(hist.astype(np.float32) - prev_hist.astype(np.float32)))

        # ────────────────────────────────────────────────────────────────────
        # STEP 4: STORE AND SMOOTH DATA
        # ────────────────────────────────────────────────────────────────────
        # Append current frame index and difference to data lists
        x_data.append(index)
        y_diff_data.append(diff)

        # Apply moving average smoothing to difference signal
        # Smoothing helps reduce noise and makes real changes more apparent
        smooth_val = smooth_signal(y_diff_data, WINDOW_SIZE)

        # Calculate first derivative (rate of change) of smoothed signal
        # High derivatives indicate scene boundaries or sudden color changes
        deriv_val = first_derivative(y_smooth_data, 1)

        # Append smoothed values and derivatives to their respective lists
        y_derivative_data.append(deriv_val)
        y_smooth_data.append(smooth_val)

        # ────────────────────────────────────────────────────────────────────
        # STEP 5: UPDATE PLOT WITH NEW DATA
        # ────────────────────────────────────────────────────────────────────
        # Update all three plot line data with new values
        line_diff.set_data(x_data, y_diff_data)
        line_smooth.set_data(x_data, y_smooth_data)
        line_deriv.set_data(x_data, y_derivative_data)

        # Recalculate plot limits to fit all data automatically
        ax.relim()
        ax.autoscale_view()
        
        # Redraw the figure with new data
        fig.canvas.draw()
        fig.canvas.flush_events()

    # ────────────────────────────────────────────────────────────────────────
    # STEP 6: PREPARE HISTOGRAM FOR VISUALIZATION
    # ────────────────────────────────────────────────────────────────────────
    # Apply Gaussian blur to smooth the histogram for better visualization
    # This reduces noise in the histogram representation
    hist_norm = cv2.GaussianBlur(hist, (5, 5), cv2.BORDER_DEFAULT)
    
    # Normalize histogram to 0-255 range for display (8-bit image)
    hist_norm = ((hist_norm * 255.0) / np.amax(hist_norm)).astype(np.uint8)
    
    # Apply JET colormap to histogram (blue=low probability, red=high probability)
    hist_display = cv2.applyColorMap(hist_norm, cv2.COLORMAP_JET)

    # Display original frame
    cv2.imshow('Frame', frame)
    
    # Display histogram of U,V chrominance channels
    cv2.imshow('Histogram (U,V Channels)', hist_display)

    # ────────────────────────────────────────────────────────────────────────
    # STEP 7: SAVE CURRENT HISTOGRAM FOR NEXT ITERATION
    # ────────────────────────────────────────────────────────────────────────
    # Store copy of current histogram for comparison with next frame
    prev_hist = hist.copy()
    
    # Increment frame counter
    index += 1

    # ────────────────────────────────────────────────────────────────────────
    # STEP 8: HANDLE KEYBOARD INPUT
    # ────────────────────────────────────────────────────────────────────────
    # Check for keyboard input with 1ms timeout
    if cv2.waitKey(1) & 0xff == 27:  # ESC key
        break

# ────────────────────────────────────────────────────────────────────────────
# CLEANUP - RELEASE RESOURCES
# ────────────────────────────────────────────────────────────────────────────
# Release the video capture object and free system resources
cap.release()

# Close all OpenCV display windows
cv2.destroyAllWindows()

# Disable interactive mode and display final plot
plt.ioff()
plt.show()
