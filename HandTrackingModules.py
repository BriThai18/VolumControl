import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, minDetection=0.5, minTrack=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.minDetection = minDetection
        self.minTrack = minTrack

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.minDetection, self.minTrack)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        # RGB colors
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)
        # print(result.multi_hand_landmarks)

        # Check if existence of hand
        if self.result.multi_hand_landmarks:
            # Loop through each hand and draw it
            for handLMS in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLMS, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        landMarkList = []

        if self.result.multi_hand_landmarks:
            #Point to particular hand
            myHand = self.result.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print(id, cx, cy)
                landMarkList.append([id, cx, cy])
                # if id == 0:
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return landMarkList

def main():
    # Frame rate
    previousTime = 0
    currentTime = 0

    cam = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        # Read frames
        success, img = cam.read()
        img = detector.findHands(img)
        landMarkList = detector.findPosition(img)
        if len(landMarkList) != 0:
            print(landMarkList[4])

        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime

        # Display fps
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 8, 255), 3)

        # Display the image
        cv2.imshow("Image", img)
        # Wait for user to press key
        cv2.waitKey(1)


if __name__ == "__main__":
    main()