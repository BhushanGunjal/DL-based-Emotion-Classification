import sounddevice as sd
from scipy.io.wavfile import write

# Sampling rate & duration
samplerate = 44100  # CD quality
duration = 5  # seconds

print("🎤 Recording... Speak happily!")
recording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=2, dtype='int16')
sd.wait()  # Wait for recording to finish

# Save the file
write("sample_audio.wav", samplerate, recording)
print("✅ Saved as sample_audio.wav")
