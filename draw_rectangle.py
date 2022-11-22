"""
顔部分を囲んで画像を表示する
コサイン類似度が0.7以上であれば「認証成功」で緑枠、そうでなければ「認証失敗」で赤枠
矩形と一緒に上記の内容を表示させる
./image/correctには「認証成功」の画像
./image/faultには「認証失敗」の画像
"""
import cv2
import dlib

def puttext(img, result, pos, color):
    cv2.putText(
        img=img, 
        text=result, 
        org=pos, 
        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
        fontScale=1,
        color=color,
        thickness=2,
        lineType=cv2.LINE_4
        )


def show_result(img_path, result, index):
    green=(0,255,0)
    red=(0,0,255)
    white=(255,255,255)
    img=cv2.imread(img_path)

    # 顔部分検出
    facedetector=dlib.get_frontal_face_detector()
    img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces=facedetector(img_gray,1)

    for face in faces:
        x1=face.left()
        y1=face.top()
        x2=face.right()
        y2=face.bottom()
        cv2.rectangle(img, (x1,y1), (x2,y2), color=green, thickness=4)        

        if(result>0.7):
            cv2.rectangle(img, (x1,y1), (x2,y2), color=green, thickness=4)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), color=green, thickness=-1)
            puttext(img=img, result="correct:"+str(result), pos=(x1+6, y2-6), color=white)

        else:
            cv2.rectangle(img, (x1,y1), (x2,y2), color=red, thickness=4)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), color=red, thickness=-1)
            puttext(img=img, result="fault:"+str(result),pos=(x1+6, y2-6), color=white)

    # if(result>0.7):
    #     print("correct")
    #     puttext(img=img, result="correct:"+str(result), pos=(x1+6, y2-6), color=white)

    # else:
    #     print("fault")
    #     puttext(img=img, result="fault:"+str(result),pos=(x1+6, y2-6), color=white)

    cv2.imwrite("./images/result_mask/"+str(index)+".png", img)
    cv2.imshow(str(index)+".png",img)
    cv2.waitKey(0)

