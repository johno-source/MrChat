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

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
        if os.path.exists('conversation_history.pkl'):
            with open('conversation_history.pkl', 'rb') as p:
                self.memory = pickle.load(p)
        else:
            self.memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)

        self._system_default = open_file('system_default.txt')
        self._user_profile = open_file('user_profile.txt')
        self._context = open_file('context.txt')



    def __call__(self, args):
        return self.generate_chat_response(args)

    def generate_chat_response(self, args):
        return fmt(self.get_conversation()({'question' : args, })['text'])

    def get_conversation(self):
        prompt = ChatPromptTemplate(
            messages=[
                self.get_system_prompt(),
                # The `variable_name` here is what must align with memory
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{question}")
            ], input_variables=['chat_history', 'question'])

        return LLMChain(llm=self.llm, prompt=prompt, verbose=True, memory=self.memory)

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
        
