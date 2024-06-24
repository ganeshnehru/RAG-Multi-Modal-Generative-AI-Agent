import pyttsx3
import time


def text_to_speech(text, chunk_size=100):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed percent (can go over 100)
    engine.setProperty('volume', 0.9)  # Volume 0-1

    # Split the text into chunks
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    for chunk in chunks:
        engine.say(chunk)
        engine.runAndWait()
        time.sleep(0.1)  # Short pause between chunks
