#!/usr/bin/env python3
from data.dataset import FILES_DIR
import yt_dlp

def download_youtube_audio(url):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{FILES_DIR}/youtube_sounds/%(title)s.%(ext)s",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "0",
        }],
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

if __name__ == "__main__":
    pass
    # download_youtube_audio("https://www.youtube.com/watch?v=d-WOGMTRyKw")
    # download_youtube_audio("https://www.youtube.com/watch?v=K6xsEng2PhU")
    # download_youtube_audio("https://www.youtube.com/watch?v=ONYd86SEisY")
    # download_youtube_audio("https://www.youtube.com/watch?v=1EdFkekr9I0")
    # download_youtube_audio("https://www.youtube.com/watch?v=0pyJL_-1s1c")
    # download_youtube_audio("https://www.youtube.com/watch?v=euEwKtP5CG4")