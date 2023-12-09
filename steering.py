import math
import keyinput
import cv2,threading
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
font = cv2.FONT_HERSHEY_SIMPLEX
# 0 For webcam input:
cap = cv2.VideoCapture(0)

def press1():
       keyinput.release_key('a')
       keyinput.release_key('d')
       keyinput.release_key('w')
       keyinput.press_key('s')
def press2():
  keyinput.release_key('s')
  keyinput.release_key('a')
  keyinput.release_key('d')
  keyinput.press_key('w')
def press3():
            keyinput.release_key('s')
            keyinput.release_key('a')
            keyinput.press_key('d')
def press4():
            keyinput.release_key('s')
            keyinput.release_key('d')
            keyinput.press_key('a')
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    image =  cv2.flip(image, 1)
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    imageHeight, imageWidth, _ = image.shape

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    co=[]
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        for point in mp_hands.HandLandmark:
           if str(point) == "HandLandmark.WRIST":
              normalizedLandmark = hand_landmarks.landmark[point]
              pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x,
                                                                                        normalizedLandmark.y,
                                                                                    imageWidth, imageHeight)

              try:
                co.append(list(pixelCoordinatesLandmark))
              except:
                  continue

    if len(co) == 2:
  
        if co[0][0] > co[1][0] and co[0][1]>co[1][1] and co[0][1] - co[1][1] > 65:

            threading.Thread(target=press4).start()
            cv2.putText(image, "Turn left", (50, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)


        elif co[1][0] > co[0][0] and co[1][1]> co[0][1] and co[1][1] - co[0][1] > 65:
       

            threading.Thread(target=press4).start()
            cv2.putText(image, "Turn left", (50, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)


        elif co[0][0] > co[1][0] and co[1][1]> co[0][1] and co[1][1] - co[0][1] > 65:
          
            threading.Thread(target=press3).start()
            cv2.putText(image, "Turn right", (50, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
 
        elif co[1][0] > co[0][0] and co[0][1]> co[1][1] and co[0][1] - co[1][1] > 65:
      

            threading.Thread(target=press3).start()
            cv2.putText(image, "Turn right", (50, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        else:
       

            threading.Thread(target=press2).start()
            cv2.putText(image, "keep straight", (50, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    if len(co)==1:

       threading.Thread(target=press1).start()
       cv2.putText(image, "keeping back", (50, 50), font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('MediaPipe Hands',image)

# Flip the image horizontally for a selfie-view display.
    if cv2.waitKey(5) & 0xFF == ord('q'):
      break
cap.release()
