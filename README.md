# Audio Transcriber

A Python pipeline that converts speech audio into text using OpenAI's Whisper model.

## What it does
- Decodes MP3/WAV audio using imageio-ffmpeg (no system FFmpeg install needed)
- Runs OpenAI's Whisper-tiny model locally via Hugging Face
- Outputs a full text transcription of the audio

## Tech Stack
- Python
- OpenAI Whisper (via Hugging Face Transformers)
- imageio-ffmpeg
- PyTorch

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Place an audio file named `sample.mp3` in the root folder
3. Run: `python app.py`

## Output
The transcribed text prints directly to the terminal.