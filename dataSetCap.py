import mss
import cv2
import numpy as np
import time

sct = mss.mss()
monitor = {"top": 250, "left": 120, "width": 1850, "height": 1200}

contador = 0
while True:
    frame = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    filename = f"yolo/dataset/images/{contador}.jpg"
    cv2.imwrite(filename, frame)
    #imprimiendo el path de la imagen
    print(f"Imagen guardada: {filename}")
    contador += 1
    time.sleep(3)  # Captura cada 1 segundo

    cv2.imshow("Detecciones", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()