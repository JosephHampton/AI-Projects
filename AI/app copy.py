from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain import PromptTemplate, LLMChain
import torch
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

tokenizer = LlamaTokenizer.from_pretrained("chavinlo/alpaca-native")

base_model = LlamaForCausalLM.from_pretrained(
    "chavinlo/alpaca-native",
    load_in_8bit=True,
    device_map='auto',
)
pipe = pipeline(
    "text-generation",
    model=base_model, 
    tokenizer=tokenizer, 
    max_length=256,
    temperature=0.6,
    top_p=0.95,
    repetition_penalty=1.2
)

local_llm = HuggingFacePipeline(pipeline=pipe)

template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction: 
{instruction}

Answer:"""

prompt = PromptTemplate(template=template, input_variables=["instruction"])

llm_chain = LLMChain(prompt=prompt, 
                     llm=local_llm
                     )

question = "What is the capital of England?"

print(llm_chain.run(question))

question = "What are alpacas? and how are they different from llamas?"

print(llm_chain.run(question))

window_memory = ConversationBufferWindowMemory(k=4)

conversation = ConversationChain(
    llm=local_llm, 
    verbose=True, 
    memory=window_memory
)
conversation.prompt.template
conversation.prompt.template = "The following is a friendly conversation between a human and an AI called Alpaca. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know."
conversation.predict(input="Hi there! I am Sam")
         
         




    

