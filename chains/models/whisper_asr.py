# import whisper
# import sounddevice as sd
# import numpy as np
# import scipy.io.wavfile as wavfile
# import webrtcvad
# import collections
#
# class WhisperASR:
#     def __init__(self, model_name="base"):
#         self.model = whisper.load_model(model_name)
#         self.vad = webrtcvad.Vad()
#         self.vad.set_mode(1)  # Set aggressiveness mode (0-3)
#
#     def record_audio(self, samplerate=16000, padding_duration_ms=2000, chunk_duration_ms=30):
#         chunk_size = int(samplerate * chunk_duration_ms / 1000)
#         num_padding_chunks = int(padding_duration_ms / chunk_duration_ms)
#         ring_buffer = collections.deque(maxlen=num_padding_chunks)
#         triggered = False
#         voiced_frames = []
#
#         try:
#             stream = sd.InputStream(samplerate=samplerate, channels=1, dtype=np.int16)
#             stream.start()
#             print("Recording...")
#
#             while True:
#                 audio_chunk, _ = stream.read(chunk_size)
#                 audio_chunk = audio_chunk[:, 0].tobytes()
#                 is_speech = self.vad.is_speech(audio_chunk, samplerate)
#
#                 if not triggered:
#                     ring_buffer.append((audio_chunk, is_speech))
#                     num_voiced = len([f for f, speech in ring_buffer if speech])
#                     if num_voiced > 0.9 * ring_buffer.maxlen:
#                         triggered = True
#                         voiced_frames.extend([f for f, s in ring_buffer])
#                         ring_buffer.clear()
#                 else:
#                     voiced_frames.append(audio_chunk)
#                     ring_buffer.append((audio_chunk, is_speech))
#                     num_unvoiced = len([f for f, speech in ring_buffer if not speech])
#                     if num_unvoiced > 0.9 * ring_buffer.maxlen:
#                         break
#
#         except Exception as e:
#             print(f"An error occurred while recording audio: {e}")
#         finally:
#             stream.stop()
#             stream.close()
#             print("Recording complete.")
#
#         audio_data = b''.join(voiced_frames)
#         audio_np = np.frombuffer(audio_data, dtype=np.int16)
#         return audio_np
#
#     def save_wav(self, file_path, audio, samplerate):
#         wavfile.write(file_path, samplerate, audio)
#
#     def transcribe_audio(self, file_path):
#         print("Transcribing...")
#         result = self.model.transcribe(file_path)
#         return result["text"]
#
#     def run(self, duration=5, samplerate=16000, file_path="recorded_audio.wav"):
#         audio = self.record_audio(samplerate=samplerate)
#         self.save_wav(file_path, audio, samplerate)
#         text = self.transcribe_audio(file_path)
#         print("Transcription:")
#         print(text)
#         return text

# if __name__ == "__main__":
#     asr_system = WhisperASR(model_name="base")
#     asr_system.run()






import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import webrtcvad
import collections

class WhisperASR:
    def __init__(self, model_name="base"):
        self.model = whisper.load_model(model_name)
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(2)  # Set aggressiveness mode (0-3)

    def record_audio(self, samplerate=16000, padding_duration_ms=2000, chunk_duration_ms=30):
        chunk_size = int(samplerate * chunk_duration_ms / 1000)
        num_padding_chunks = int(padding_duration_ms / chunk_duration_ms)
        ring_buffer = collections.deque(maxlen=num_padding_chunks)
        triggered = False
        voiced_frames = []

        try:
            stream = sd.InputStream(samplerate=samplerate, channels=1, dtype=np.int16)
            stream.start()
            print("Recording...")

            while True:
                audio_chunk, _ = stream.read(chunk_size)
                audio_chunk = audio_chunk[:, 0].tobytes()
                is_speech = self.vad.is_speech(audio_chunk, samplerate)

                if not triggered:
                    ring_buffer.append((audio_chunk, is_speech))
                    num_voiced = len([f for f, speech in ring_buffer if speech])
                    if num_voiced > 0.9 * ring_buffer.maxlen:
                        triggered = True
                        voiced_frames.extend([f for f, s in ring_buffer])
                        ring_buffer.clear()
                else:
                    voiced_frames.append(audio_chunk)
                    ring_buffer.append((audio_chunk, is_speech))
                    num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                    if num_unvoiced > 0.9 * ring_buffer.maxlen:
                        break

        except Exception as e:
            print(f"An error occurred while recording audio: {e}")
        finally:
            stream.stop()
            stream.close()
            print("Recording complete.")

        audio_data = b''.join(voiced_frames)
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        return audio_np

    def save_wav(self, file_path, audio, samplerate):
        wavfile.write(file_path, samplerate, audio)

    def transcribe_audio(self, file_path):
        print("Transcribing...")
        result = self.model.transcribe(file_path)
        return result["text"]

    def run(self, duration=5, samplerate=16000, file_path="recorded_audio.wav"):
        audio = self.record_audio(samplerate=samplerate)
        self.save_wav(file_path, audio, samplerate)
        text = self.transcribe_audio(file_path)
        print("Transcription:")
        print(text)
        return text
