import nemo
from nemo.collections.asr.models import EncDecCTCModelBPE
import os
from sys import byteorder
from array import array
from struct import pack
import pyaudio
import wave

class NeMoASR:
    def __init__(self, model_name="stt_en_conformer_ctc_xlarge"):
        self.asr_model = EncDecCTCModelBPE.from_pretrained(model_name=model_name)
        self.THRESHOLD = 500
        self.CHUNK_SIZE = 1024
        self.FORMAT = pyaudio.paInt16
        self.RATE = 44100
        self.SILENT_CHUNKS = int(3 * self.RATE / self.CHUNK_SIZE)  # Number of chunks of silence for 3 seconds

    def is_silent(self, snd_data):
        """Returns 'True' if below the 'silent' threshold"""
        return max(snd_data) < self.THRESHOLD

    def normalize(self, snd_data):
        """Average the volume out"""
        MAXIMUM = 16384
        multiplier = float(MAXIMUM) / max(abs(i) for i in snd_data)
        return array('h', [int(i * multiplier) for i in snd_data])

    def trim(self, snd_data):
        """Trim the blank spots at the start and end"""
        def _trim(snd_data):
            snd_started = False
            trimmed_data = array('h')
            for i in snd_data:
                if not snd_started and abs(i) > self.THRESHOLD:
                    snd_started = True
                    trimmed_data.append(i)
                elif snd_started:
                    trimmed_data.append(i)
            return trimmed_data

        snd_data = _trim(snd_data)
        snd_data.reverse()
        snd_data = _trim(snd_data)
        snd_data.reverse()
        return snd_data

    def add_silence(self, snd_data, seconds):
        """Add silence to the start and end of 'snd_data' of length 'seconds' (float)"""
        silence = [0] * int(seconds * self.RATE)
        return array('h', silence) + snd_data + array('h', silence)

    def record(self):
        """
        Record audio from the microphone and return the data as an array of signed shorts.
        Normalizes the audio, trims silence from the start and end, and pads with 0.5 seconds of silence.
        """
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT, channels=1, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK_SIZE)

        num_silent = 0
        snd_started = False
        recorded_data = array('h')

        while True:
            snd_data = array('h', stream.read(self.CHUNK_SIZE, exception_on_overflow=False))
            if byteorder == 'big':
                snd_data.byteswap()
            recorded_data.extend(snd_data)

            if self.is_silent(snd_data):
                if snd_started:
                    num_silent += 1
            else:
                snd_started = True
                num_silent = 0

            if snd_started and num_silent > self.SILENT_CHUNKS:
                break

        sample_width = p.get_sample_size(self.FORMAT)
        stream.stop_stream()
        stream.close()
        p.terminate()

        recorded_data = self.normalize(recorded_data)
        recorded_data = self.trim(recorded_data)
        recorded_data = self.add_silence(recorded_data, 0.5)
        return sample_width, recorded_data

    def record_to_file(self, path):
        """Records from the microphone and outputs the resulting data to 'path'"""
        sample_width, data = self.record()
        data = pack('<' + ('h' * len(data)), *data)

        with wave.open(path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(sample_width)
            wf.setframerate(self.RATE)
            wf.writeframes(data)

    def transcribe_audio(self, file_path):
        """Transcribe the given audio file using the ASR model"""
        transcriptions = self.asr_model.transcribe(paths2audio_files=[file_path])
        return transcriptions[0]

if __name__ == '__main__':
    asr_recorder = NeMoASR()
    print("Please speak a word into the microphone.")
    audio_file = 'demo.wav'
    asr_recorder.record_to_file(audio_file)
    print(f"Done - result written to {audio_file}.")

    transcription = asr_recorder.transcribe_audio(audio_file)
    print(f"Audio in {audio_file} was recognized as: {transcription}")
