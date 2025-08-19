from ultralytics import YOLO
import mss
import cv2
import numpy as np

# Capturador
sct = mss.mss()

#model = YOLO("yolo11n.pt").to("cuda")
model = YOLO("runs/detect/train8/weights/best.pt").to("cuda")

# Definir región (x, y, ancho, alto)
monitor = {"top": 250, "left": 100, "width": 1850, "height": 1200}

while True:
    frame = np.array(sct.grab(monitor))  # Captura
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Convertir a BGR para OpenCV

    results = model.predict(frame, conf=0.4, device="cuda")

    # Dibujar resultados directamente en la imagen
    annotated = results[0].plot()

    cv2.imshow("Detecciones", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()