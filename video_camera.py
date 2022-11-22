#カメラから画像取得

import cv2
import dlib

# save_dir="./images/videocap_submit/submit.png"
save_dir="./images/videocap_verify/verify1.png"

cap=cv2.VideoCapture(0)

# 写真撮影
# while(1):
#     ret, frame=cap.read()
#     if ret:
#         cv2.imwrite(save_dir, frame)
    
#     key=cv2.waitKey(1)
#     if(key==27 or key==13): break

# 顔検出
detector=dlib.get_frontal_face_detector()

while(1):
    ret, frame=cap.read()
    img_gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=detector(img_gray,1)
    # for face in faces:
    #     x1=face.left()
    #     y1=face.top()
    #     x2=face.right()
    #     y2=face.bottom()
    #     cv2.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,0), thickness=4)
    # cv2.imshow("face", frame)
    # k=cv2.waitKey(1)
    # if(k==27 or k==13):
    #     break
    if ret:
        cv2.imwrite(save_dir, frame)
        break
    
cap.release()
cv2.destroyAllWindows()
