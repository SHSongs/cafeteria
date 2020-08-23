import cv2
import imutils

haar_upper_body_cascade = cv2.CascadeClassifier("data/haarcascade_upperbody.xml")
full_body_cascade = cv2.CascadeClassifier("data/haarcascade_fullbody.xml")
full_body_cascade = cv2.CascadeClassifier("data/haarcascade_fullbody.xml")

# Uncomment this for real-time webcam detection
# If you have more than one webcam & your 1st/original webcam is occupied,
# you may increase the parameter to 1 or respectively to detect with other webcams, depending on which one you wanna use.

# video_capture = cv2.VideoCapture(0)

# For real-time sample video detection
video_capture = cv2.VideoCapture("subway.mp4")
video_width = video_capture.get(3)
video_height = video_capture.get(4)
fps = video_capture.get(cv2.CAP_PROP_FPS) # 또는 cap.get(5)
fourcc = cv2.VideoWriter_fourcc(*'DIVX') # 코덱 정의

out = cv2.VideoWriter('otter_out.avi', fourcc, fps, (int(video_width), int(video_height))) # VideoWriter 객체 정의


while True:
    ret, frame = video_capture.read()

    # Bframe = imutils.resize(frame, width=1000)  # resize original video for better viewing performance
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert video to grayscale

    upper_body = haar_upper_body_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 100),  # Min size for valid detection, changes according to video size or body size in the video.
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    full_body = full_body_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 100),  # Min size for valid detection, changes according to video size or body size in the video.
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the upper bodies
    for (x, y, w, h) in upper_body:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0),
                      1)  # creates green color rectangle with a thickness size of 1
        cv2.putText(frame, "Upper Body Detected", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                    2)  # creates green color text with text size of 0.5 & thickness size of 2
    cv2.imshow('Video', frame)  # Display video

    # Draw a rectangle around the upper bodies
    for (x, y, w, h) in full_body:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0),
                      1)  # creates green color rectangle with a thickness size of 1
        cv2.putText(frame, "full Body Detected", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                    1)  # creates green color text with text size of 0.5 & thickness size of 2
    cv2.imshow('Video', frame)  # Display video

    out.write(frame)
    # stop script when "q" key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release capture
out.release()
video_capture.release()
cv2.destroyAllWindows()