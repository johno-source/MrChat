#
# This cognition node is responsible for augmenting the chat ledger with any input information from outside of the chatbot.
#
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
from utils import *
from chat_ledger import ChatLedger
from langchain_groq import ChatGroq

class ContextCognitionNode:
    def __init__(self, model_name="llama3-70b-8192", prompt_file="../prompts/ExtractContextPrompt1.txt"):
        prompt_text = open_file(prompt_file)
        human = "{text}"
        prompt = ChatPromptTemplate.from_messages([("system", prompt_text), ("human", human)])
        output_parser = JsonOutputParser()
        model = ChatGroq(model_name=model_name)
        self._chain = prompt | model | RunnableLambda(FilterOutExtraToJSON) | output_parser

    def run(self, ledger):
        resp = self._chain.invoke(ledger.get_user_input())

        return resp['thought']
        