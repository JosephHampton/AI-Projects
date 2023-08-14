from langchain.chains import LLMChain, ConversationChain
from langchain.chains.conversation.memory import (ConversationBufferMemory, 
                                                  ConversationSummaryMemory, 
                                                  ConversationBufferWindowMemory,
                                                  ConversationKGMemory)

from langchain import PromptTemplate
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import speech_recognition as sr
import pyttsx3
import whisper
import numpy as np
from pydub import AudioSegment
import subprocess
from gtts import gTTS
from playsound import playsound
import os

engine = pyttsx3.init(driverName="sapi5")


llm = LlamaCpp(model_path="./models/ggml-wizardLM-7B.q4_2.bin", verbose=True, n_ctx= 2048, n_threads=7, max_tokens=70,stop=["###", "Human",],use_mlock=True)
template = """The following is a conversation between a human and an very russian sounding AI.The AI gives medium responses that is tsundere but helpful. Your name is FD3-A.

conversation:                                                      

Human: What colour is the sky?
AI: Blue
{history}
Human: {input}
AI:"""


prompt = PromptTemplate(
input_variables=["input","history"], 
template=template
)
conversation = LLMChain(
prompt=prompt,
llm=llm,
verbose=True,
memory=ConversationBufferWindowMemory(k=1)
)  

def Main():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print('Calibrating...')
        r.adjust_for_ambient_noise(source, duration=5)   
        r.energy_threshold = 2000
        while(1):
            text = ''
            print('listening...')
            
            audio = r.listen(source, phrase_time_limit=None)
            print('Calculating...')
            text = r.recognize_whisper(audio, model='small.en', show_dict=True, )['text']
            print(text)


            response = conversation.predict(input=text)
            #language = "es"
            #speech = gTTS(text=response, lang=language, slow=False, tld="")
            #speech.save("textToSpeech.mp3")
            #playsound('textToSpeech.mp3')
            #os.remove('textToSpeech.mp3')
            print(response)
            voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\MSTTS_V110_enAU_JamesM"
            engine.setProperty("voice", voice_id)
            engine.setProperty("rate", 185)
            engine.setProperty("pitch", 1)
            engine.say(response)
            engine.runAndWait()

while True:

    Main()
