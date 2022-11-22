# yt-dlp -f b "動画のURL"
from yt_dlp import YoutubeDL

ydl_opts={"format": "best"}
urls=[
    #url
    ]

with YoutubeDL(ydl_opts) as ydl:
    for i in range(len(urls)):
        print(urls[i])
        result=ydl.download(url_list=urls[i])
        print(result)