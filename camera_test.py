
import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera could not be opened.")
else:
    print("Camera opened successfully.")

    ret, frame = cap.read()

    if ret:
        print("Camera frame captured successfully.")
        cv2.imshow("Camera Test", frame)
        cv2.waitKey(0)
    else:
        print("Camera opened, but no frame was received.")

cap.release()
cv2.destroyAllWindows()
