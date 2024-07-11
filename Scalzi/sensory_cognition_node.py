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
from scalzi_logger import scalzi_logger
from capabilities import search_web, take_screenshot, capture_webcam, read_clipboard

class SensoryCognitionNode:
    def __init__(self, model_name="llama3-70b-8192", prompt_file="../prompts/SelectFunctionPrompt1.txt"):
        prompt_text = open_file(prompt_file)
        human = "{text}"
        prompt = ChatPromptTemplate.from_messages([("system", prompt_text), ("human", human)])
        output_parser = JsonOutputParser()
        model = ChatGroq(model_name=model_name)
        self._chain = prompt | model | RunnableLambda(FilterOutExtraToJSON) | output_parser

    def run(self, ledger):
        scalzi_logger.info(f'Running Sensory Cognition Node: {ledger.get_user_input()}')
        resp = self._chain.invoke(ledger.get_user_input())
        func = remove_quotes(resp["function"]).lower()
        scalzi_logger.info(f'Sensory Cognition Chain thought: {resp["thought"]} function: {func}')

        if func == 'search web':
            sensory_input = self.search_web(resp["search"], ledger)
        elif func == 'capture webcam':
            sensory_input = self.capture_webcam()
        elif func == 'take screenshot':
            sensory_input = self.take_screenshot()
        elif func == 'read clipboard':
            sensory_input = self.read_clipboard()
        else:
            sensory_input = "None"

        ledger.set_sensory_inputs(func, resp['thought'], resp['search'], sensory_input)
        scalzi_logger.info(f'Sensory Cognition Result: {sensory_input}')

        return sensory_input

    def capture_webcam(self):
        return f'Capture webcam not implemented'
        
    def take_screenshot(self):
        return f'Screen shot not implemented'

    def read_clipboard(self):
        return 'read clipboard not implemented.'

    def search_web(self, search_terms, ledger):
        search_str,result = search_web(search_terms)
        ledger.set_search_terms(search_str)
        ledger.set_search_results(result)

        result_str = result[0]['content']
        scalzi_logger.info(f'Search Web Result: {result_str}')
        return result_str