import speech_recognition as sr
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
r = sr.Recognizer()


def get_large_audio_transcription(path):
    sound = AudioSegment.from_wav(path)
    # split audio sound where silence is 700 miliseconds or more and get chunks
    chunks = split_on_silence(sound,
                              # experiment with this value for your target audio file
                              min_silence_len=500,
                              # adjust this per requirement
                              silence_thresh=sound.dBFS - 14,
                              # keep the silence for 1 second, adjustable as well
                              keep_silence=500,
                              )
    folder_name = "audio-chunks"
    # create a directory to store the audio chunks
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_text = ""
    # process each chunk
    for i, audio_chunk in enumerate(chunks, start=1):
        # export audio chunk and save it in
        # the `folder_name` directory.
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        # recognize the chunk
        with sr.AudioFile(chunk_filename) as source:
            audio_listened = r.record(source)
            # try converting it to text
            try:
                text = r.recognize_google(audio_listened)
            except sr.UnknownValueError as e:
                print("")
            else:
                text = f"{text.capitalize()}. "
                whole_text += text
    # return the text for all chunks detected
    return whole_text





def OfflineVideoTranscript(path):
    import moviepy.editor as mp
    my_clip = mp.VideoFileClip(path)
    print(my_clip)
    my_clip.audio.write_audiofile(r"C:/Users/vijayalakshmi/PycharmProjects/offline_vid_to_aud/res.wav")
    path = "C:/Users/vijayalakshmi/PycharmProjects/offline_vid_to_aud/res.wav"
    summary = get_large_audio_transcription(path)
    return summary






