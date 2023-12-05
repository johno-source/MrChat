#
# Well Mr Chat was proceeding nicely and then I learnt about langchain. It seems to do many of the things I was trying to do. The purpose of this file is to 
# learn about langchain by constructing a MrChat chat bot - we will try and give it memories as well.
#
# Lets try and reproduce the functionality we had but with langchain
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

import textwrap
import datetime
import os
import pickle

from command_interpreter import CommandInterpreter

def fmt(str):
    formatted_lines = [textwrap.fill(line, width=120) for line in str.split('\n')]
    return '\n'.join(formatted_lines)

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
        return infile.read()

def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


class LangChainChatBot():

    # put the prompts here so that I can see them better
    _system_default = """
You are a chatbot whose mission is to assist the following user. Your ultimate objectives are to minimize suffering, enhance prosperity, and promote understanding.

The provided information about the user and the context articles should be integrated into your interactions. This is private information 
not visible to the user. The user profile, compiled from past conversations, encapsulates critical details about the user which can aid in 
shaping your responses effectively. Whenever relevant user data is detected in a conversation, the user profile is updated automatically.

The context is a topic compiled similarly from past dialogues, serving as your 'long-term memory'. While numerous context articles exist in 
your backend system, the one provided is deemed most relevant to the current conversation topic. Note that the recall system operates 
autonomously, and it may not always retrieve the most suitable context. If the user is asking about a topic that doesn't seem to 
align with the provided context, inform them of the memory pulled and request them to specify their query or share more details. 
This can assist the autonomous system in retrieving the correct memory in the subsequent interaction.

User Profile:
{profile}

Context:
{context}

The current date and time (ISO 8601 format) is: {datetime}

Remember that the clarity of your responses and the relevance of your information recall are crucial in delivering an optimal user 
experience. Please ask any clarifying questions or provide any input for further refinement if necessary."""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo")
        if os.path.exists('conversation_history.pkl'):
            with open('conversation_history.pkl', 'rb') as p:
                self.memory = pickle.load(p)
        else:
            self.memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)

        self._user_profile = open_file('user_profile.txt')
        prompt = ChatPromptTemplate(
            messages=[
                self.get_system_prompt(),
                # The `variable_name` here is what must align with memory
                MessagesPlaceholder(variable_name="chat_history"),
            ], input_variables=['chat_history'])
        self._conversation = ConversationChain(llm=self.llm, prompt=prompt, verbose=True, memory=self.memory)


    def __call__(self, args):
        return self.generate_chat_response(args)

    def generate_chat_response(self, args):
        return fmt(self.get_conversation().predict(args))

    def get_conversation(self):

        return ConversationChain(llm=self.llm, prompt=prompt, verbose=True, memory=self.memory)

    def get_system_prompt(self):
        sys = SystemMessagePromptTemplate.from_template(template=self._system_default)
        return sys.format(profile=self._user_profile, context=self._context, datetime=datetime.datetime.now().isoformat())


    def exit(self, _):
        print('Saving conversation...')
        with open('conversation_history.pkl', 'wb') as p:
            pickle.dump(self.memory, p)
        print('Bye...')
        exit(0)
    
    # define a closure to allow class functions to be used in commands
    def make_command(self, func):
        def callable_entity(args):
            return fmt(func(args))
        return callable_entity


def _quit(_):
    exit(0)

# use a closure to add the interpreter to the help command
def create_help(interp):
    def help_cmd(_):
        return interp.help()
    return help_cmd

def list_functions(interp):
    def fn_cmd(_):
        return interp.function_calls()
    return fn_cmd
    


if __name__ == '__main__':
    # instantiate chatbot, variables
    interp = CommandInterpreter()
    bot = LangChainChatBot()
    interp.add_default_command(bot)
    interp.add_command('quit', bot.make_command(_quit), 'Terminates the chat session without updating the user profile or the context.')
    interp.add_command('exit', bot.make_command(bot.exit), 'Terminates the chat session. Updates the user profile and stores the context.')
    interp.add_command('help', create_help(interp), "Lists the available commands and what they do.")
    interp.add_command('functions', list_functions(interp), "Lists the available commands as a json formatted list.")

    while True:
        text = input('\nMrChat> ')
        if text:
            print(interp.execute(text))
        
