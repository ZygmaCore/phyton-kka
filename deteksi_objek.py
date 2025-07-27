import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

# Buka kamera laptop
kamera = cv2.VideoCapture(0)

print("Tekan 'q' untuk keluar dari program.")

while True:
    status, frame = kamera.read()
    if not status:
        print("Gagal membuka kamera.")
        break

    # Deteksi objek
    kotak, label, conf = cv.detect_common_objects(frame)

    # Gambar bounding box dan label
    output = draw_bbox(frame, kotak, label, conf)

    # Tampilkan hasil
    cv2.imshow("Deteksi Objek", output)

    # Jika tekan tombol 'q', keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()