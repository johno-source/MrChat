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

class SensoryCognitionNode:
    def __init__(self, model_name="llama3-70b-8192", prompt_file="../prompts/SelectFunctionPrompt1.txt"):
        prompt_text = open_file(prompt_file)
        human = "{text}"
        prompt = ChatPromptTemplate.from_messages([("system", prompt_text), ("human", human)])
        output_parser = JsonOutputParser()
        model = ChatGroq(model_name=model_name)
        self._chain = prompt | model | RunnableLambda(FilterOutExtraToJSON) | output_parser

    def run(self, ledger):
        resp = self._chain.invoke(ledger.get_user_input())
        func = remove_quotes(resp["function"]).lower()

        if func == 'search web':
            sensory_input = self.search_web(resp["search"])
        elif func == 'capture webcam':
            sensory_input = self.capture_webcam()
        elif func == 'take screenshot':
            sensory_input = self.take_screenshot()
        elif func == 'read clipboard':
            sensory_input = self.read_clipboard()
        else:
            sensory_input = "None"

        ledger.set_sensory_inputs(func, resp['thought'], resp['search'], sensory_input)

        return sensory_input

    def search_web(self, keys):
        return f'Search Web not implmented: {keys}'

    def capture_webcam(self):
        return f'Capture webcam not implemented'
        
    def take_screenshot(self):
        return f'Screen shot not implemented'

    def read_clipboard(self):
        return 'read clipboard not implemented.'
        