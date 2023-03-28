# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai
import os
import sounddevice as sd
from scipy.io.wavfile import write
from dotenv import load_dotenv


def record_audio(filename="output.wav"):
    fs = 22050  # Sample rate
    seconds = 6  # Duration of recording
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    print("\nBegin recording...")
    sd.wait()  # Wait until recording is finished
    print("End recording")
    write(filename, fs, myrecording)  # Save as WAV file


def transcribe_audio(filename="output.wav"):
    audio_file = open(filename, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    print("Ich habe folgendes verstanden:")
    print(transcript.text)
    return transcript.text


def query_chatgpt(prompt):
    messages = []
    messages.append(
        {"role": "system", "content": "Be precise, be concise, be polite and positive."})

    message = prompt
    messages.append({"role": "user", "content": message})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages)
    reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": reply})
    print("\n" + reply + "\n")
    return reply


def main():
    os.system("clear")
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    soundfile_name = "output.wav"

    print("Hallo ich bin der Awesomebot vom CityLAB Berlin!")

    record_audio(soundfile_name)
    prompt = transcribe_audio(soundfile_name)
    reply = query_chatgpt(prompt)

    os.system("say -r180 " + reply)


if __name__ == '__main__':
    main()
