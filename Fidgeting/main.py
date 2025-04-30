# import cv2
# import mediapipe as mp
# import time
# import math as math


# class HandTrackingDynamic:
#     def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
#         self.__mode__   =  mode
#         self.__maxHands__   =  maxHands
#         self.__detectionCon__   =   detectionCon
#         self.__trackCon__   =   trackCon
#         self.handsMp = mp.solutions.hands
#         self.hands = self.handsMp.Hands()
#         self.mpDraw= mp.solutions.drawing_utils
#         self.tipIds = [4, 8, 12, 16, 20]

#     def findFingers(self, frame, draw=True):
#         imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         self.results = self.hands.process(imgRGB)  
#         if self.results.multi_hand_landmarks: 
#             for handLms in self.results.multi_hand_landmarks:
#                 if draw:
#                     self.mpDraw.draw_landmarks(frame, handLms,self.handsMp.HAND_CONNECTIONS)

#         return frame

#     def findPosition( self, frame, handNo=0, draw=True):
#         xList =[]
#         yList =[]
#         bbox = []
#         self.lmsList=[]
#         if self.results.multi_hand_landmarks:
#             myHand = self.results.multi_hand_landmarks[handNo]
#             for id, lm in enumerate(myHand.landmark):
            
#                 h, w, c = frame.shape
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 xList.append(cx)
#                 yList.append(cy)
#                 self.lmsList.append([id, cx, cy])
#                 if draw:
#                     cv2.circle(frame,  (cx, cy), 5, (255, 0, 255), cv2.FILLED)

#             xmin, xmax = min(xList), max(xList)
#             ymin, ymax = min(yList), max(yList)
#             bbox = xmin, ymin, xmax, ymax
#             print( "Hands Keypoint")
#             print(bbox)
#             if draw:
#                 cv2.rectangle(frame, (xmin - 20, ymin - 20),(xmax + 20, ymax + 20),
#                                (0, 255 , 0) , 2)

#         return self.lmsList, bbox
    
#     def findFingerUp(self):
#          fingers=[]

#          if self.lmsList[self.tipIds[0]][1] > self.lmsList[self.tipIds[0]-1][1]:
#               fingers.append(1)
#          else:
#               fingers.append(0)

#          for id in range(1, 5):            
#               if self.lmsList[self.tipIds[id]][2] < self.lmsList[self.tipIds[id]-2][2]:
#                    fingers.append(1)
#               else:
#                    fingers.append(0)
        
#          return fingers

#     def findDistance(self, p1, p2, frame, draw= True, r=15, t=3):
         
#         x1 , y1 = self.lmsList[p1][1:]
#         x2, y2 = self.lmsList[p2][1:]
#         cx , cy = (x1+x2)//2 , (y1 + y2)//2

#         if draw:
#               cv2.line(frame,(x1, y1),(x2,y2) ,(255,0,255), t)
#               cv2.circle(frame,(x1,y1),r,(255,0,255),cv2.FILLED)
#               cv2.circle(frame,(x2,y2),r, (255,0,0),cv2.FILLED)
#               cv2.circle(frame,(cx,cy), r,(0,0.255),cv2.FILLED)
#         len= math.hypot(x2-x1,y2-y1)

#         return len, frame , [x1, y1, x2, y2, cx, cy]

# def main():
        
#         ctime=0
#         ptime=0
#         cap = cv2.VideoCapture(0)
#         detector = HandTrackingDynamic()
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#         if not cap.isOpened():
#             print("Cannot open camera")
#             exit()

#         while True:
#             ret, frame = cap.read()

#             frame = detector.findFingers(frame)
#             lmsList = detector.findPosition(frame)
#             # if len(lmsList)!=0:
#             #     #print(lmsList[0])

#             ctime = time.time()
#             fps =1/(ctime-ptime)
#             ptime = ctime

#             cv2.putText(frame, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
 
#     #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             cv2.imshow('frame', frame)
#             cv2.waitKey(1)


                
# if __name__ == "__main__":
#             main()

import cv2
import numpy as np
import mediapipe as mp
from one_euro_filter import OneEuroFilter
from collections import deque

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)

freq = 30.0
min_cutoff = 2.5
beta = 0.4
d_cutoff = 1.0

filters = {
    'Left': [
        [OneEuroFilter(freq, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff) for _ in range(21)],
        [OneEuroFilter(freq, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff) for _ in range(21)]
    ],
    'Right': [
        [OneEuroFilter(freq, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff) for _ in range(21)],
        [OneEuroFilter(freq, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff) for _ in range(21)]
    ]
}

tip_ids = [
    mp_hands.HandLandmark.THUMB_TIP,
    mp_hands.HandLandmark.INDEX_FINGER_TIP,
    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    mp_hands.HandLandmark.RING_FINGER_TIP,
    mp_hands.HandLandmark.PINKY_TIP
]

window_len = 30
finger_dir_thr = 5
finger_speed_thr = 2.0
pinch_dist_thr = 40.0
pinch_evt_thr = 3

pos_deques = {h: {t: deque(maxlen=window_len) for t in tip_ids} for h in ("Left","Right")}
vel_deques = {h: {t: deque(maxlen=window_len) for t in tip_ids} for h in ("Left","Right")}
pinch_dist = {h: deque(maxlen=window_len) for h in ("Left","Right")}
pinch_events = {h: deque(maxlen=window_len) for h in ("Left","Right")}
last_pinch = {h: False for h in ("Left","Right")}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(img_rgb)
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w, _ = img.shape

    if res.multi_hand_landmarks:
        for lmks, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
            label = handedness.classification[0].label
            for idx, lm in enumerate(lmks.landmark):
                x_raw, y_raw = lm.x * w, lm.y * h
                x_f = filters[label][0][idx](x_raw)
                y_f = filters[label][1][idx](y_raw)
                lm.x = x_f / w
                lm.y = y_f / h

            for tip in tip_ids:
                lm = lmks.landmark[tip]
                pos = (lm.x * w, lm.y * h)
                dq = pos_deques[label][tip]
                if dq:
                    prev = dq[-1]
                    vel = (pos[0] - prev[0], pos[1] - prev[1])
                    vel_deques[label][tip].append(vel)
                dq.append(pos)

            lm_i = lmks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            lm_t = lmks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            d = np.hypot((lm_i.x - lm_t.x) * w, (lm_i.y - lm_t.y) * h)
            pd = pinch_dist[label]
            pd.append(d)
            cur_pinch = d < pinch_dist_thr
            if cur_pinch and not last_pinch[label]:
                pinch_events[label].append(1)
            last_pinch[label] = cur_pinch

            f_f = False
            for tip in tip_ids:
                vq = vel_deques[label][tip]
                if len(vq) == window_len:
                    dir_changes = sum(1 for i in range(1, window_len) if
                                      vq[i][0] * vq[i-1][0] + vq[i][1] * vq[i-1][1] < 0)
                    speeds = [np.hypot(v[0], v[1]) for v in vq]
                    if dir_changes > finger_dir_thr and np.mean(speeds) > finger_speed_thr:
                        f_f = True
                        break

            f_i = sum(pinch_events[label]) > pinch_evt_thr

            if f_f or f_i:
                cx, cy = int(lm_i.x * w), int(lm_i.y * h)
                cv2.putText(img, f"{label} FIDGETING", (cx, cy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            mp_drawing.draw_landmarks(
                img, lmks, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

    cv2.imshow("Fidget Detection", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()





