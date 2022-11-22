"""
# 顔認証は学習済みモデルの"facenet"を使用する
# 入力:画像、出力:512個の数値でこの数値で比較対象とのコサイン類似度で認証する
# マスクなしを登録して、マスク着用の画像を認証する > 0.792305、認証できた
"""

from cropped_face import image2vec
from similarity_func import cos_similarity
from draw_rectangle import show_result

# 登録する画像:"r"、見つけたい人
r_path="./images/videocap_submit/submit.png"
ptest=image2vec(img_path=r_path, mode="r")

# 認証する画像:"v"、見つけた人
v_path="./images/videocap_verify/verify5.png"

# 顔部分の検出
print(v_path)
verify=image2vec(img_path=v_path, mode="v")

# 類似度を計算して顔認証
print("[類似度]")
print(v_path,":",cos_similarity(verify, ptest))
# 画面に表示
show_result(img_path=v_path, result=round(cos_similarity(verify, ptest),3), index=40)


