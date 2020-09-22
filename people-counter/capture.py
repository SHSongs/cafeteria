import cv2
import time
import requests

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

url = "http://10.120.72.140:5000/capture"

headers = {'Content-Type': 'application/json'}

cnt = 1

while True:
    ret, frame = capture.read()
    cv2.imshow("VideoFrame", frame)

    i_name = "img" + str(cnt) + ".jpg"

    cv2.imwrite(i_name, frame)
    time.sleep(5)
    files = {'image': open(i_name, 'rb')}
    response = requests.request("POST", url, files=files, headers=headers)

    print(response.text)

    cnt += 1

    if cv2.waitKey(1) > 0:
        break

capture.release()
cv2.destroyAllWindows()