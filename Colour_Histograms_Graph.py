import cv2
import numpy as np
import matplotlib.pyplot as plt

WINDOW_SIZE = 5

# --- Configuração do Matplotlib para Tempo Real ---
plt.ion()
fig, ax = plt.subplots()

# Duas linhas: uma para diff, outra para smooth
line_diff, = ax.plot([], [], 'r-', label='Diff', alpha=0.5)
line_smooth, = ax.plot([], [], 'b-', label='Smooth', linewidth=2)
line_deriv, = ax.plot([], [], 'g-', label='Deriv', linewidth=2)

ax.set_title("Diferença de Histograma ao Longo do Tempo")
ax.set_xlabel("Frame")
ax.set_ylabel("Diff (Soma Absoluta)")
ax.legend()

# Listas para armazenar os dados
x_data = []
y_diff_data = []
y_smooth_data = []
y_derivative_data = []

# --- Seu código original ---
cap = cv2.VideoCapture("../Projet2_Videos/Extrait5-Matrix-Helicopter_Scene(280p).m4v")
prev_hist = None
index = 0

def smooth_signal(signal, window_size):
    """Suaviza o sinal usando média móvel"""
    if len(signal) < window_size:
        return signal[-1]  # Retorna o último valor se não há dados suficientes
    kernel = np.ones(window_size) / window_size
    return np.convolve(signal, kernel, mode='valid')[-1]

def first_derivative(signal, time_interval):
    """Calcula a primeira derivada usando diferenças finitas"""
    if len(signal) < 2:
        return 0  # Retorna 0 se não há dados suficientes
    return (signal[-1] - signal[-2]) / time_interval

while True:
    ret, frame = cap.read()
    if not ret:
        break

    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    hist = cv2.calcHist([yuv], [1, 2], None, [256, 256], [0, 256, 0, 256])

    if prev_hist is not None:
        diff = np.sum(np.abs(hist.astype(np.float32) - prev_hist.astype(np.float32)))

        x_data.append(index)
        y_diff_data.append(diff)

        # Smooth calculado sobre o histórico completo de diffs
        smooth_val = smooth_signal(y_diff_data, WINDOW_SIZE)

        deriv_val = first_derivative(y_smooth_data, 1)

        y_derivative_data.append(deriv_val)
        y_smooth_data.append(smooth_val)

        # Atualiza as duas linhas independentemente
        line_diff.set_data(x_data, y_diff_data)
        line_smooth.set_data(x_data, y_smooth_data)
        line_deriv.set_data(x_data, y_derivative_data)

        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

    hist_norm = cv2.GaussianBlur(hist, (5, 5), cv2.BORDER_DEFAULT)
    hist_norm = ((hist_norm * 255.0) / np.amax(hist_norm)).astype(np.uint8)
    hist_display = cv2.applyColorMap(hist_norm, cv2.COLORMAP_JET)

    cv2.imshow('Image', frame)
    cv2.imshow('Histogramme (u,v)', hist_display)

    prev_hist = hist.copy()
    index += 1

    if cv2.waitKey(1) & 0xff == 27:
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()