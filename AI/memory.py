from langchain.chains import LLMChain, ConversationChain
from langchain.chains.conversation.memory import (ConversationBufferMemory, 
                                                  ConversationSummaryMemory, 
                                                  ConversationBufferWindowMemory,
                                                  ConversationKGMemory)
from langchain.llms import GPT4All
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

local_path = "./models/ggml-wizardLM-7B.q4_2.bin"
llm = GPT4All(model=local_path, n_ctx=500,n_threads=8,n_predict=100, verbose=True)

conversation_with_summary = ConversationChain(
    llm=llm, 
    # We set a low k=2, to only keep the last 2 interactions in memory
    memory=ConversationBufferWindowMemory(k=2), 
    verbose=True
)
conversation_with_summary.predict(input="Do you like dogs?")
conversation_with_summary.predict(input="why do you like them?")
