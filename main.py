# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai
from os import system
import sounddevice as sd
from scipy.io.wavfile import write

system("clear")

# RECORD AUDIO

print("Hallo ich bin der Awesomebot vom CityLAB Berlin!")

fs = 22050  # Sample rate
seconds = 6  # Duration of recording
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
print("\nBegin recording...")
sd.wait()  # Wait until recording is finished
print("End recording")
write('output.wav', fs, myrecording)  # Save as WAV file

# TRANSCRIBE AUDIO VIA WHISPER

openai.api_key = "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"

audio_file = open("output.wav", "rb")
transcript = openai.Audio.transcribe("whisper-1", audio_file)
print("Ich habe folgendes verstanden:")
print(transcript.text)

# OUTPUT ANSWER FROM CHATGPT

messages = []
messages.append(
    {"role": "system", "content": "The reply should be a transformation of the sentence into a very bureaucratic and formal language. keep it short but complex."})

message = transcript.text
messages.append({"role": "user", "content": message})
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages)
reply = response["choices"][0]["message"]["content"]
messages.append({"role": "assistant", "content": reply})
print("\n" + reply + "\n")
system("say -r180 -v 'Yannick' '" + reply+"'")
