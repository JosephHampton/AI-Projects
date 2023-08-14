import torch
from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms.base import LLM
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from langchain import HuggingFacePipeline
import os

import torch


os.environ["CUDA_VISIBLE_DEVICES"]="0"


model_id = 'google/flan-t5-small'# go for a smaller model if you dont have the VRAM
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to('cuda')


pipe = pipeline(
    "text2text-generation",
    model=model, 
    tokenizer=tokenizer, 
    max_length=100
)

local_llm = HuggingFacePipeline(pipeline=pipe)

print(local_llm('What is the capital of England? '))
