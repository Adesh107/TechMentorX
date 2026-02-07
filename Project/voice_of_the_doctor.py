import os
from gtts import gTTS
from elevenlabs import ElevenLabs, save

ELEVENLABS_API_KEY = "sk_efa361b1d87a658b1e9d685db3c24c55a1c45211a47b6cb2"


def text_to_speech_with_gtts(input_text, output_filepath):
    """Convert text to speech using gTTS (free)."""
    audioobj = gTTS(text=input_text, lang="en", slow=False)
    audioobj.save(output_filepath)
    return output_filepath


def text_to_speech_with_elevenlabs(input_text, output_filepath):
    """Convert text to speech using ElevenLabs (premium quality)."""
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    
    audio = client.text_to_speech.convert(
        text=input_text,
        voice_id="EXAVITQu4vr4xnSDxMaL",
        model_id="eleven_turbo_v2",
        output_format="mp3_22050_32"
    )
    
    save(audio, output_filepath)
    return output_filepath