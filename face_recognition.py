"""
# 顔認証は学習済みモデルの"facenet"を使用する
# 入力:画像、出力:512個の数値でこの数値で比較対象とのコサイン類似度で認証する
"""

from cropped_face import image2vec
from similarity_func import cos_similarity
from draw_rectangle import show_result
import glob

# 登録する画像:"r"、見つけたい人
t=1
r_path="./images/submit/*.png"
register_img=sorted(glob.glob(r_path))
r=[]
for reg in register_img:
    r.append(reg)
ptest=image2vec(img_path=r[t], mode="r")

# 認証する画像:"v"、見つけた人
v_path="./images/verify_img/*.png"
verify_img=sorted(glob.glob(v_path))
v=[]
for ver in verify_img:
    v.append(ver)

# 顔部分の検出
p=[]
for i in range(len(v)):
    print(v[i])
    verify=image2vec(img_path=v[i], mode="v")
    p.append(verify)

# 類似度を計算して顔認証
for i in range(len(v)):
    print("[類似度]")
    print(v[i],":",cos_similarity(p[i], ptest))
    # 画面に表示
    idx=t*100
    show_result(img_path=v[i], result=round(cos_similarity(p[i], ptest),3), index=idx+i)


