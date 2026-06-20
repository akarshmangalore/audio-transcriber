import imageio_ffmpeg
import os
import numpy as np
import subprocess

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import torch
from transformers import pipeline, BartForConditionalGeneration, BartTokenizer

def load_audio_with_imageio_ffmpeg(file_path):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, "-i", file_path,
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ar", "16000",
        "-ac", "1",
        "pipe:1"
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    return audio

def transcribe_audio(audio_file_path):
    print(" Loading Whisper model...")
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f" Decoding audio: {audio_file_path}...")
    audio_array = load_audio_with_imageio_ffmpeg(audio_file_path)
    print(" Transcribing...")
    result = pipe(
        {"array": audio_array, "sampling_rate": 16000},
        return_timestamps=True
    )
    return result["text"]

def summarize_text(text):
    print("\n Summarising...")
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    if len(text.split()) < 30:
        return "Text too short to summarise."
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=60,
        min_length=15,
        length_penalty=2.0,
        num_beams=4
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    sample_audio = "sample.mp3"
    try:
        text_output = transcribe_audio(sample_audio)
        print("\n Transcription Result:")
        print("-" * 40)
        print(text_output)
        print("-" * 40)

        summary_output = summarize_text(text_output)
        print("\n Summary:")
        print("-" * 40)
        print(summary_output)
        print("-" * 40)
    except Exception as e:
        print(f"\n Error: {e}")
