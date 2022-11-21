"""
読み込んだ画像から顔部分を切り抜いて、512個の数値に変換
入力:画像
出力:要素数が512のndarray型ベクトル
"""
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# 画像から顔検出するAIの定義
mtcnn=MTCNN(image_size=160, margin=10)
# 512の数値に変換するAIの定義
resnet=InceptionResnetV1(pretrained="vggface2").eval()

def image2vec(img_path, mode):
    # 画像読み込み
    img=Image.open(img_path)

    # 画像から顔検出して、512個の数値に変換
    # 登録モード0のときは、./images/register_imgに保存する
    if(mode=="r"):
        img_cropped=mtcnn(img, save_path="./images/register_img/Registered.png")
    else:
        img_cropped=mtcnn(img)

    img_embedding=resnet(img_cropped.unsqueeze(0))

    # tensor型からnumpy型に変換
    p=img_embedding.squeeze().to('cpu').detach().numpy().copy()

    return p
    
