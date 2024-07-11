#
# This module captures the state of the chat bot. I have chosen to not use langchain for the conversation elements so that we have more flexibility.
#

class ChatLedger:
    def __init__(self):
        self.previous_subject = "None"
        self.current_subject = "None"
        self.conversation_summary = ""
        self.tool_output = ""
        self.conversation = []
        self.user_input = ""
        self.chat_output = ""
        self.search_terms = ''
        self.search_results = []

    def set_user_input(self, text):
        self.user_input = text
        return text

    def get_user_input(self):
        return self.user_input

    def set_chat_output(self, text):
        self.chat_output = text
        return text

    def get_chat_output(self):
        return self.chat_output

    def set_sensory_inputs(self, stimulus, thought, args, sensory_input):
        self.stimulus = stimulus
        self.stimulus_thought = thought
        self.stimulus_args = args
        self.sensory_input = sensory_input

    def set_search_terms(self, search_terms):
        self.search_terms = search_terms

    def set_search_results(self, search_results):
        self.search_results = search_results


    

