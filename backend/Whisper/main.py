from openai import OpenAI
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate 
client = OpenAI(
    api_key=""
)
audio_file = open("sampleAudio.mp3", "rb")
 

# Step 1: Transcribe Hindi audio
transcription = client.audio.transcriptions.create(
    model="whisper-1",  # or gpt-4o-transcribe
    file=audio_file,
    response_format="text",
    language="hi"  # Detect Hindi
)

hindi_text = transcription
print("Hindi (Devanagari):", hindi_text)

# Step 2: Convert Hindi (Devanagari) to Hinglish (Roman letters)
hinglish_text = transliterate(hindi_text, sanscript.DEVANAGARI, sanscript.ITRANS)
print("Hinglish (Roman Hindi):", hinglish_text)