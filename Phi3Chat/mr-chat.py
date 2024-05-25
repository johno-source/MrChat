#
#                                                                           Synopsis
# This is my first attempt to make a local chatbot using ollama and llama3 instruct. I am also trying to implement a memory recall system using agents.
# I started by looking at llama_index but have recently pivoted back to langchain and a library new to me, called langgraph.
# My aim is to make this a modular base that will allow new agents and functions to be added as we go.
#

import textwrap
import json
from time import sleep
from command_interpreter import CommandInterpreter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import Ollama
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder
import pickle as pkl
import os


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
        return infile.read()
    
def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def fmt(str):
    formatted_lines = [textwrap.fill(line, width=120) for line in str.split('\n')]
    return '\n'.join(formatted_lines)


class LocalChatBot():

    def __init__(self):
        self._history = self._restore_history()
        system_default = open_file('system_default.txt')
        update_user_profile = open_file('system_update_user_profile.txt')
        update_context = open_file('system_update_context.txt')
        self._user_profile = open_file('user_profile.txt')
        self._context = open_file('context.txt')
        chat_prompt = ChatPromptTemplate.from_messages([ 
            ('system', system_default ),
            MessagesPlaceholder(variable_name="chat_history"),
            ('human', '{input}' ),
            ])

        local_llm = 'phi3:14b-medium-4k-instruct-q8_0'
        chat_llm = ChatOllama(model=local_llm, temperature=0, base_url="http://192.168.86.2:11434", keep_alive=-1)
        self._chat_chain = chat_prompt | chat_llm
        
        update_profile = PromptTemplate(input_variables=['history', 'UPD'], template=update_user_profile)
        completion_llm = Ollama(model=local_llm, temperature=0, base_url="http://192.168.86.2:11434", keep_alive=-1)
        self._update_profile_chain = update_profile | completion_llm | StrOutputParser()

        update_context = PromptTemplate(input_variables=['history', 'context'], template=update_context)
        self._update_context_chain = update_context | completion_llm | StrOutputParser()


    def __call__(self, args):
        return self.generate_chat_response(args)

    def generate_chat_response(self, text):
        response = self._chat_chain.invoke({'chat_history': self._history.messages, 'profile': self._user_profile, 'context': self._context, 'input': text})
        self._history.add_user_message(text)
        self._history.add_ai_message(response.content)

        self._prompt_eval_count = response.response_metadata['prompt_eval_count']
        self._eval_count = response.response_metadata['eval_count']
        return fmt(response.content)

    def get_stats(self, _):
        return f'Prompt Eval Count: {self._prompt_eval_count}\nEval Count: {self._eval_count}'


    def history(self, _):
        formatted_lines = []
        for m in self._history.messages:
            formatted_lines.append(f'{m.type.upper()}: {m.content}')
        return '\n'.join(formatted_lines)
    
    def _restore_history(self):
        if os.path.exists('message_history.pkl'):
            with open('message_history.pkl', 'rb') as f:
                return pkl.load(f)
        else:
            return ChatMessageHistory()
        
    def _save_history(self):
        with open('message_history.pkl', 'wb') as f:
            pkl.dump(self._history, f)

    def clear_history(self, _):
        self._history = ChatMessageHistory()
        self._save_history()
        return 'Cleared!'
    
    def user_profile(self, _):
        return self._user_profile
    
    def current_context(self, _):
        return self._context

    def update_user_profile(self, _):
        response = self._update_profile_chain.invoke({'history': self.history(' '), 'UPD': self._user_profile})
        self._user_profile = response
        return response

    def update_context(self, _):
        response = self._update_context_chain.invoke({'history': self.history(' '), 'context': self._context})
        self._context = response
        return response
    
    def exit(self, _):
        self._save_history()
        self.update_context('NA')
        self.update_user_profile('NA')
        save_file("context.txt", self._context)
        save_file("user_profile.txt", self._user_profile)
        print('Bye!')
        exit(0)

    
    # define a closure to allow class functions to be used in commands
    def make_command(self, func):
        def callable_entity(args):
            return fmt(func(args))
        return callable_entity

if __name__ == '__main__':
    # instantiate chatbot, variables
    interp = CommandInterpreter()
    bot = LocalChatBot()
    interp.add_default_command(bot)
    interp.add_command('exit', bot.make_command(bot.exit))
    interp.add_command('history', bot.make_command(bot.history))
    interp.add_command('user', bot.make_command(bot.user_profile))
    interp.add_command('context', bot.make_command(bot.current_context))
    interp.add_command('update_user', bot.make_command(bot.update_user_profile))
    interp.add_command('update_context', bot.make_command(bot.update_context))
    interp.add_command('get_stats', bot.make_command(bot.get_stats))
    interp.add_command('clear_history', bot.make_command(bot.clear_history))

    while True:
        text = input('\nMrChat> ')
        if text:
            print(interp.execute(text))
        
