
import random
import wave
from time import sleep
from django.contrib.sessions.backends.base import SessionBase

def sample_cut(session:SessionBase, in_file,out_file):
    session["status"] = {"status":"running","name":"Cutting","description":"Opening Cutting file.."}
    session.save()
    with wave.open(str(in_file), 'rb') as wav:
        params = wav.getparams()
        frame_rate = params.framerate
        n_frames = params.nframes
        duration = n_frames / frame_rate

        if duration <= 3:
            raise ValueError("Input file is shorter than 3 seconds.")

        start_time = random.uniform(0, duration - 3)
        start_frame = int(start_time * frame_rate)
        end_frame = start_frame + int(3 * frame_rate)

        wav.setpos(start_frame)
        frames = wav.readframes(end_frame - start_frame)
    # sample sleep to make it longer

    sleep(1.5)
    session["status"] = {"status":"running","name":"Cutting","description":"Closing Cutting file.."}
    session.save()
    with wave.open(str(out_file), 'wb') as out_wav:
        out_wav.setparams(params)
        out_wav.writeframes(frames)

    # sample sleep to make it longer
    sleep(1.2)
    return int(start_time * 1000)


def process(session:SessionBase, in_file,out_file):
    session["status"] = {"status":"running","name":"Loading File","description":"Loading the sound file.."}
    session.save()
    # start process of loading file
    # start process of clustering/predicting
    # start 