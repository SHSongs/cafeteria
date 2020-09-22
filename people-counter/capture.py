import cv2
import time

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = capture.read()
    cv2.imshow("VideoFrame", frame)

    cv2.imwrite("img.jpg", frame)
    time.sleep(5)

    if cv2.waitKey(1) > 0:
        break

capture.release()
cv2.destroyAllWindows()