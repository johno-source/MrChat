#
#                                                                           Synopsis
# With the publishing of phi3 it becomes feasible to implement a reflective chatbot. This code is intended to implement many of the ideas I was playing with in the 
# OpenAI space but, as it is run locally I am hoping to have a significant speed boost and to be able to do many more parses of the data. I am not sure it will be
# needed but having a 128K window may also alleviate the need to filter the input data as heavily.
#

import textwrap
import json
from time import sleep
from command_interpreter import CommandInterpreter
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
        return infile.read()
    
def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

def fmt(str):
    formatted_lines = [textwrap.fill(line, width=120) for line in str.split('\n')]
    return '\n'.join(formatted_lines)

def fmt_conversation(conversation):
    formatted_lines = []
    for c in conversation:
        formatted_lines.append(f'Role: {c["role"]}')
        formatted_lines.append(f'Content: {c["content"]}')
    return '\n'.join(formatted_lines)


class Phi3ChatBot():

    def __init__(self):
        self._history = list()
        self._system_default = open_file('system_default.txt')
        self._user_profile = open_file('user_profile.txt')
        self._context = open_file('context.txt')
        self._update_user_profile = open_file('system_update_user_profile.txt')
        self._update_context = open_file('system_update_context.txt')
        self._system_message = self._system_default.replace('<<PROFILE>>', self._user_profile).replace('<<CONTEXT>>', self._context)
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct", 
            device_map="cuda", 
            torch_dtype="auto", 
            trust_remote_code=True, 
            attn_implementation="eager",
        )
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

        self.generation_args = {
            "max_new_tokens": 4000,
            "return_full_text": False,
            "do_sample": False,
        }

    def __call__(self, args):
        return self.generate_chat_response(self.compose_conversation(args))

    def compose_conversation(self, text):
        # continue with composing conversation and response
        self._history.append({'role': 'user', 'content': text})
        conversation = list()
        self._system_message = self._system_default.replace('<<PROFILE>>', self._user_profile).replace('<<CONTEXT>>', self._context)
        conversation.append({'role': 'user', 'content': self._system_message})
        conversation += self._history
        return conversation

    def generate_chat_response(self, conversation):

        print(fmt_conversation(conversation))

        # generate a response
        output = self.pipe(conversation, **self.generation_args)
        response = output[0]['generated_text']
        self._history.append({'role': 'assistant', 'content': response})
        return fmt(response)


    def history(self, _):
        return fmt_conversation(self._history)
    
    def system_prompt(self, _):
        return self._system_message
    
    def user_profile(self, _):
        return self._user_profile
    
    def current_context(self, _):
        return self._context
    
    def update_user_profile(self, _):
        conversation = list()
        update_profile = self._update_user_profile.replace('<<UPD>>', self._user_profile)
        conversation.append({'role': 'user', 'content': update_profile})
        conversation += self._history[:-1]

        print(fmt_conversation(conversation))

        output = self.pipe(conversation, **self.generation_args)
        response = output[0]['generated_text']
        print(response)
        self._user_profile = response
        return response

    def update_context(self, _):
        conversation = list()
        conversation += self._history
        update_context = self._update_context.replace('<<CONTEXT>>', self._context)
        conversation.append({'role': 'system', 'content': update_context})
        output = self.pipe(conversation, **self.generation_args)
        response = output[0]['generated_text']
        self._context = response
        return response
    
    def exit(self, _):
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
    bot = Phi3ChatBot()
    interp.add_default_command(bot)
    interp.add_command('exit', bot.make_command(bot.exit))
    interp.add_command('history', bot.make_command(bot.history))
    interp.add_command('system', bot.make_command(bot.system_prompt))
    interp.add_command('user', bot.make_command(bot.user_profile))
    interp.add_command('context', bot.make_command(bot.current_context))
    interp.add_command('update_user', bot.make_command(bot.update_user_profile))
    interp.add_command('update_context', bot.make_command(bot.update_context))

    while True:
        text = input('\nMrChat> ')
        if text:
            print(interp.execute(text))
        
