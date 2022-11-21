# 顔認証は学習済みモデルの"facenet"を使用する
# 入力：画像、出力：512個の数値でこの数値で比較対象とのコサイン類似度で認証する
# リアルタイムで判定
from cropped_face import image2vec
from similarity_func import cos_similarity
from draw_rectangle import show_result
import cv2
import glob

# 登録する:"r"
r_path="./images/submit/*.png"
register_img=glob.glob(r_path)
r=[]
for reg in register_img:
    r.append(reg)

# 認証する:"v"
cap=cv2.VideoCapture(0)
save_dir="./images/live/input_cap.png"
while(1):
    success, img=cap.read()
    cv2.imwrite(save_dir, img)

    # 顔を比較する
    p=image2vec(img_path=save_dir, mode="v")
    ptest=image2vec(img_path=r[0], mode="r")

    # 類似度を計算して顔認証
    imgtest=cos_similarity(p,ptest)

    cv2.imshow("Face recognition", img)
    k=cv2.waitKey(1)
    if(k==27 or k==13):
        break

cap.release()
cv2.destroyAllWindows()
# 出力結果
# print("test:", imgtest)

# 画面に表示
# show_result(img_path=v[0], result=img1test)
