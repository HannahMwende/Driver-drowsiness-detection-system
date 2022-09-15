import cv2
import numpy as np
from PIL import Image, ImageDraw
import face_recognition
from tensorflow import keras
import tensorflow as tf
from playsound import playsound

eye_model = keras.models.load_model('final_model1.h5')



# webcam frame is inputted into function

def eye_cropper(frame):

    # create a variable for the facial feature coordinates

    facial_features_list = face_recognition.face_landmarks(frame)


    # create a placeholder list for the eye coordinates
    # and append coordinates for eyes to list unless eyes
    # weren't found by facial recognition

    try:
        eye = facial_features_list[0]['left_eye']
    except:
        try:
            eye = facial_features_list[0]['right_eye']
        except:
            return


    # establish the max x and y coordinates of the eye

    x_max = max([coordinate[0] for coordinate in eye])
    x_min = min([coordinate[0] for coordinate in eye])
    y_max = max([coordinate[1] for coordinate in eye])
    y_min = min([coordinate[1] for coordinate in eye])


    # establish the range of x and y coordinates

    x_range = x_max - x_min
    y_range = y_max - y_min


      # in order to make sure the full eye is captured,
      # calculate the coordinates of a square that has a
      # 50% cushion added to the axis with a larger range and
      # then match the smaller range to the cushioned larger range

    if x_range > y_range:
        right = round(.5*x_range) + x_max
        left = x_min - round(.5*x_range)
        bottom = round((((right-left) - y_range))/2) + y_max
        top = y_min - round((((right-left) - y_range))/2)
    else:
        bottom = round(.5*y_range) + y_max
        top = y_min - round(.5*y_range)
        right = round((((bottom-top) - x_range))/2) + x_max
        left = x_min - round((((bottom-top) - x_range))/2)


    # crop the image according to the coordinates determined above

    cropped = frame[top:(bottom + 1), left:(right + 1)]

    # resize the image

    cropped = cv2.resize(cropped, (80,80))
    image_for_prediction = cropped.reshape(-1, 80, 80, 3)


    return image_for_prediction


# initiate webcam

cap = cv2.VideoCapture(0)
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(cap.get(cv2.CAP_PROP_FPS))

if not cap.isOpened():
    raise IOError('Cannot open webcam')
counter = 0

# create a while loop that runs while webcam is in use

while True:

    #creating a while loop that runs when webcam is in use

    ret, frame = cap.read()

    # use only every other frame to manage speed and memory usage

    frame_count = 0
    if frame_count == 0:
        frame_count += 1
        pass
    else:
        count = 0
        continue

    # function called on the frame

    image_for_prediction = eye_cropper(frame)
    try:
        image_for_prediction = image_for_prediction/255.0
    except:
        continue

    # get prediction from model

    prediction = eye_model.predict(image_for_prediction)

    # Based on prediction, display either "Open Eyes" or "Closed Eyes"

    if prediction < 0.5:
        counter = 0
        status = 'Open'
        cv2.putText(frame, status, (round(w/2)-80,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2, cv2.LINE_4)

    else:
        counter = counter + 1
        status = 'Closed'
        cv2.putText(frame, status, (round(w/2)-104,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_4)
       
        # if the counter is greater than 3, play and show alert that user is asleep

        if counter > 2:
            cv2.putText(frame, 'DRIVER SLEEPING', (round(w/2)-136,round(h) - 146), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_4)
            cv2.imshow('Drowsiness Detection', frame)
            k = cv2.waitKey(1)
            playsound('rooster.mov')
            counter = 1
            continue
    cv2.imshow('Drowsiness Detection', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()