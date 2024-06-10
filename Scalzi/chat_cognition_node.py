#
#                                                                           Synopsis
# This module is used to implement the chat cognition node. It takes all the info in the ledger and uses it to produce the output.
#

from langchain_core.prompts import ChatPromptTemplate
from utils import *
from langchain_groq import ChatGroq


class ChatCognitionNode():

    def __init__(self, model_name = "llama3-70b-8192", sys_prompt_file="../prompts/JarvisPrompt.txt"):
        system_prompt = open_file(sys_prompt_file)

        chat_prompt = ChatPromptTemplate.from_messages([ 
            ('system', system_prompt ),
            # MessagesPlaceholder(variable_name="chat_history"),
            ('human', '{input}' ),
            ])
        chat_llm = ChatGroq(model_name=model_name)
        self._chat_chain = chat_prompt | chat_llm

    def run(self, ledger):
        response = self._chat_chain.invoke({'input': ledger.get_user_input()})
        ledger.set_chat_output(response.content)
        return response.content

