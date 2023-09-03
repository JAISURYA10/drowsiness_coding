import cv2
import time
import pygame

pygame.init()

alert_sound = pygame.mixer.Sound('C:\\Users\\jaisu\\PycharmProjects\\pythonProject4\\beep-04.wav')

eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

closed_time_start = None
closed_time_threshold = 5
sound_playing = False
sound_start_time = None
sound_duration = 5

alert_message = ""
alert_message_start_time = None
alert_duration = 3

eye_boxes_disappeared_time = None
eye_boxes_disappeared_threshold = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eyeCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

    is_heavy_rain = True
    low_visibility = True

    if len(eyes) == 0:
        if is_heavy_rain and low_visibility:
            if closed_time_start is None:
                closed_time_start = time.time()
            else:
                if time.time() - closed_time_start >= closed_time_threshold and not sound_playing:
                    alert_sound.play()
                    sound_playing = True
                    sound_start_time = time.time()
                    alert_message = "   ALERT"
                    alert_message_start_time = time.time()

            if eye_boxes_disappeared_time is None:
                eye_boxes_disappeared_time = time.time()
            elif time.time() - eye_boxes_disappeared_time >= eye_boxes_disappeared_threshold:
                alert_sound.play()
                eye_boxes_disappeared_time = None
                alert_message_start_time = time.time()

        if sound_playing and time.time() - sound_start_time >= sound_duration:
            alert_sound.stop()
            sound_playing = False

        if alert_message and time.time() - alert_message_start_time >= alert_duration:
            alert_message = ""
        frame = frame

    else:
        closed_time_start = None
        if sound_playing:
            alert_sound.stop()
            sound_playing = False
        eye_boxes_disappeared_time = None

    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        eye_midpoint = (ex + ew // 2, ey + eh // 2)

        horizontal_ratio = ew / eh
        vertical_ratio = eh / ew

        status = "unknown"

        if horizontal_ratio < 0.25 and vertical_ratio < 0.25:  # Adjust the threshold as needed
            status = "eyes closed"
        else:
            status = "eyes open"

        cv2.putText(frame, status, (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    weather_text = "Weather: "
    if is_heavy_rain:
        weather_text += "Heavy Rain, "
    if low_visibility:
        weather_text += "Low Visibility"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 255), -1)
    cv2.putText(frame, weather_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if alert_message:
        text_size = cv2.getTextSize(alert_message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = (frame.shape[0] + text_size[1]) // 2
        cv2.putText(frame, alert_message, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Eyes Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()