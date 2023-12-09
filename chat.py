import random
import json
import os
import wikipedia as wp
import webbrowser
import torch,time
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import pyautogui
import HandTrackingModule as htm
import cv2
import time
import math
import numpy as np
import HandTracking as htrack
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import math
import keyinput
import cv2,threading
import mediapipe as mp
       
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Luna"
###########################
wCam, hCam = 640, 480
wScr, hScr = pyautogui.size()
frameR = 150
smoothing = 5
plocX, plocY = 0, 0
clocX, clocY = 0, 0
##########################

detector = htm.HandDetector()

brflag = False

def testDevice():
   cap = cv2.VideoCapture(0) 
   if cap is None or not cap.isOpened():
       return True
       
def Mouse(img):
    global frameR, smoothing, plocX, plocY, clocX, clocY, wScr, wCam, hScr, hCam
    # finding hands
    detector.findhands(img)
    lmlist, bbox = detector.findPosition(img)

    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

    # 2. get the tip of index and midel finger
    if len(lmlist) != 0:
        Xindex, Yindex = lmlist[8][1], lmlist[8][2]
        # 3. check which one is up?
        fingers = detector.fingersUp()
        # 4. index: moving mode
        if fingers[1] == 1 and fingers[2] == 0:
            # 5. coordinates the position (cam: 640*480) to (screen: 2560 × 1600)
            xMOUSE = np.interp(Xindex, (frameR, wCam - frameR), (0, wScr))
            yMOUSE = np.interp(Yindex, (frameR, hCam - frameR), (0, hScr))
            # 6. smoothen value
            clocX = plocX + (xMOUSE - plocX) / smoothing
            clocY = plocY + (yMOUSE - plocY) / smoothing
            # 7. move mouse
            pyautogui.moveTo(clocX, clocY, duration=0.1)
            cv2.circle(img, (Xindex, Yindex), 15, (20, 180, 90), cv2.FILLED)
            plocY, plocX = clocY, clocX

        # 8. both are up: clicking mode
        if fingers[1] == 1 and fingers[2] == 1:
            # 9. finding distance
            length, bbox = detector.findDistance(8, 12, img)
            # 10. click if distance was short
            if length > 40:
                pyautogui.click()
                time.sleep(5)

    return img

def main():
    global brflag
    if testDevice():
       brflag=True

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
    brflag=False
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)

        img = Mouse(img)
        if brflag:
            break

        # 11. display
        cv2.imshow("result", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
threading.Thread(target=main).start()
def open_camera():
      os.system('start microsoft.windows.camera:')
def open_mail():{
       os.system('start outlookmail:')   
}
def search_wiki(msg):
      return wp.summary(msg.split('search')[-1], sentences = 2)

def steering():
       global brflag
       mp_drawing = mp.solutions.drawing_utils
       if testDevice():
           brflag=True
       mp_drawing_styles = mp.solutions.drawing_styles
       mp_hands = mp.solutions.hands
       font = cv2.FONT_HERSHEY_SIMPLEX
       # 0 For webcam input:
       cap = cv2.VideoCapture(0)
       brflag = False

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
       threading.Thread(target=main).start()

def startgame():
    threading.Thread(target=steering).start()
    os.system('''cd "Crazy Cars" && gtlauncher.exe''')
    
    
def chvol():

    #initializing window width & height
    camWidth = 840
    camHeight = 640
    if testDevice():
       brflag=True
    #capturing the video stream from the camera
    frame = cv2.VideoCapture(0)
    frame.set(3,camWidth)
    frame.set(4,camHeight)
    ptime = 0 #presentTime
    brflag=False



    #create object for our HandTracking class
    detector = htrack.handTracking(detect_conf=0.7)


    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    #volume.GetMute()
    volume.GetMasterVolumeLevel()
    volRange = volume.GetVolumeRange()
     #value of volumne as 0 ---> 100 (as system vol value)
    # value (-10) as 51, so more the lesser values(in negative) greater is our system vol value.
    volbar = 400
    barPer = 0
    vol = 0
    minVol = volRange[0]
    maxVol = volRange[1]
    #vol range = -65 to 0


    #################################################

    while True:
        success,img = frame.read()
        handImage = detector.detectHands(img)
        lmList = detector.FindPosition(handImage,draw=False) #lmList has all the values for 21 different points in form of (x,y) coordinates
        #print("detection value points as x,y",lmList)
        #print("totol no of detections we are able to get as:",len(lmList))
        if len(lmList)!=0:

            x1,y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            cx, cy = (x1+x2)//2, (y1+y2)//2

            cv2.circle(handImage,(x1,y1),10,(255,0,255),cv2.FILLED)
            cv2.circle(handImage,(x2,y2),10,(255,0,255),cv2.FILLED)
            cv2.line(handImage,(x1,y1),(x2,y2),(255,0,255),3)
            cv2.circle(handImage,(cx,cy),10,(255,0,255),cv2.FILLED)


            length = math.hypot(x2-x1,y2-y1)
            #hand range is taken as 50-200. it can vary according the to focal length of camera.
            vol = np.interp(length,[25,200],[minVol,maxVol])
            #print("hi i am volume",vol)
            barPer = np.interp(length,[25,200],[0,100])
            volbar = np.interp(length, [25, 200], [400, 150])
            volume.SetMasterVolumeLevel(vol, None)

            if length < 50:
                cv2.circle(handImage,(cx,cy),10,(255,0,0),cv2.FILLED)


        cv2.rectangle(handImage,(50,150),(85,400),(255,0,0),3)
        cv2.rectangle(handImage,(50,int(volbar)),(85,400),(255,102,102),cv2.FILLED)
        cv2.putText(handImage,f'{int(barPer)}%',(40,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),3)
        #calculate Frame per second value
        ctime = time.time()
        FPS = 1/(ctime - ptime)
        ptime = ctime

        cv2.imshow("Volume control",img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    threading.Thread(target=main).start()
                
def get_response(msg):
    import webbrowser
    command = msg.lower()
    if msg=="Listening...":
        return None
    
    if 'chrome' in msg.lower():
        webbrowser.open('www.google.com')
        return "Opened"
    elif 'youtube' in msg.lower():
        webbrowser.open('www.youtube.com')
        return "Opened"
    elif 'camera' in command:
        open_camera()
        return "Opened"
    elif 'mail' in command:
        open_mail()
        return "Opened"
    elif "music" in msg.lower():
        l = os.listdir('./music')
        f = ''
        lt = ['mp3','wav','avi']
        for i in l:
            if i.split('.')[-1].lower() in lt:
                f = os.getcwd()+'/music/'+i
                break
        if f=='':
            return 'No audio file found'
        else:
            os.popen(f)
            return 'playing...'
    elif "volume" in msg.lower():
           chvol()
           return "Done"
    elif "game" in msg.lower():
           startgame()
           return "Enjoy The Game"
    elif "search" in command or 'wiki' in command:
        return wp.summary(msg.split('search')[-1], sentences = 2)
    
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
            if prob.item() > 0.75:
                 for intent in intents['intents']:
                      if tag == intent["tag"]:
                           open_camera()
                           return "Opened"
                      if prob.item() > 0.75:
                          for intent in intents['intents']:
                              if tag == intent["tag"]:
                                  open_mail()
                                  return "Opened"
                              if tag == intent["tag"]:
                                  search_wiki(msg)
                
    return "I do not understand..."

    
