import cv2
import mediapipe as mp
import numpy as np

# 모델 경로
model_path = r"path"

knn = cv2.ml.KNearest_create()
knn = knn.load(model_path)

# 사용할 손 감지 모델 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

mp_drawing = mp.solutions.drawing_utils

# 카메라 불러오기 
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img_rgb)

    # 손 마디마다의 각도 구하기
    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            joint_v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] 
            joint_v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] 

            v = joint_v2 - joint_v1 

            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) 

            angle = np.degrees(angle)

            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 9)
            idx = str(int(results[0][0])) # 결과값

    else:
        idx = None

    cv2.putText(img, idx, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,0), 3)
    cv2.imshow('Hand Gesture Recognition', img)

    # ESC를 눌러 종료하기
    if cv2.waitKey(5) & 0xFF == 27:
        break

# 종료료
cap.release()
cv2.destroyAllWindows()
