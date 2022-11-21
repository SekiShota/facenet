# 類似度を求める関数、コサイン類似度
import numpy as np

def cos_similarity(p1, p2):
    cos_s=np.dot(p1,p2)/(np.linalg.norm(p1)*np.linalg.norm(p2))
    return cos_s