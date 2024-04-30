#
#                                                                           Synopsis
# Inspired by David Shapiro's Reflective Journalling tool and REMO (Rolling Episodic Memory Organiser) I want to realise my vision of a Memory Recall Chat bot.
# This is meant to be a Chatbot that can demonstrate an ability to learn by forming memories.
#
# Background
# Both of David's tools only aim to save and recall parts of the previous conversation. While I see this has value I think there is more value in distilling the conversation into
# the above categories. The source conversation will also be stored but only be recalled via reference from the higher level abstractions of information. 
# The reflective chatbot make use of filenames to capture summaries that are presented to ChatGPT to allow it to decide which one to recall. This is a powerful idea but I doubt that it
# will scale well as pretty soon all of the context window will be taken up with filenames. The limit will only be a couple of hundred memories or maybe 1000 if the 32k version of GPT is used. 
# REMO is a very interesting piece of code. It uses Google's Universal Sentence Encoder (USE) to create a 512 vector of searchable memories. Cosine Similarity is then used to hone in on the
# most appropriate memory. Interestingly the whole system is implememted using YAML files and a file system. No use is made of the filenames. The key idea behind REMO, however, is the
# use of clusters and summaries. Rather than using each conversation piece the conversation is paired with the piece before it. Although it is never stated I believe this is meant to
# be an approximation of context. The two are summarized and the USE encoding is found of the result. K-Means clustering is performed periodically on the USE encodings to permit similar
# conversation fragments to be grouped together. A summary of the cluster is then formed and its USE encoding is calculated again. This process is continued for as many levels as 
# necessary, making sure there is only ever roughly 10 items in each cluster. Searching then becomes a process of finding the most relevant top level cluster and then drilling down until
# an endpoint is found. All of the summaries above it are kept to give a taxonomy for the conversation fragment. REMO does not try to put this into a chatbot so we do not know how well
# it will work. The exciting thing is the ability to scale it to many memories, with a log10 scale of the level. ( A billion memories would take 9 levels, and only about 90 comparisons to 
# find a specific memory. )
#
# Goals
# Both of David's modules focus on memory recall. I believe that memory formation is just as important. Assuming a text interface it is my proposal to use gpt 3.5 or 4.0 with prompts
# to extract memories from the text. There are multiple types of memories that could be used here. There are memories that are:
#
#   1. General factoids
#   2. Episodic factoids
#   3. Reflective thoughts
#   4. Profile memories (facts about particular named entities)
# 
# I want to maintain a card on each of these. These should be able to be searched quickly and a REMO style approach would allow for the memories to be of a massive scale. 
# My aim is to have a chat bot that can answer the following questions:
#
#   1. What did we talk about yesterday?
#   2. What are my children's names?
#   3. What developments happened in my research areas this week?
#   4. What did I say about electric vehicles the other day?
#
# The first question means that the chatbot needs to know the current date and time. This can be achieved by including it in the prompt. This will give it a sense of now. It also
# requires it to be able to search memories based on a timestamp. To do this we would like a chatbot that can make callbacks. 
#
# We need the prompt to include a context card. The context includes the current time, the person being spoken to, chatbot persona, the current topic if there is one. 
# There is possibly other info to include here but it can be expanded as we go. The person being spoken to and the chatbot persona both have named entity profiles. 
# The system prompt will not be static but rather include all these profiles. At some point each of these profiles should be updated with any new information gained from the text. 
#
# There is only ever really one thing we want the chatbot to do that is not responding to the user - which is recalling memories. This makes the callback simpler.

import openai
import textwrap
import json
from time import sleep
from command_interpreter import CommandInterpreter


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
        return infile.read()
    
# define the chatbot interface to memory recall
chatbot_functions = list()
chatbot_functions.append({'name' : 'entity', 'description': "Finds information about a named entity.", 'parameters': {
    'type': 'object', 'properties': { 
        'named_entity' : { 
            'type': 'string',
            'description': 'The entity to find information about.'
            }
        }}})

def find_entity(named_entity):
    print(f'\nQuery about: {named_entity}\n')
    return { 'role' : 'function', 'name' : 'entity', 'content' : f'No information known about {named_entity}' }

def unknown_function(name):
    print(f'\nUnknown function call: {name}\n')
    return { 'role' : 'function', 'name' : name, 'content' : f'Unknown function: {name}' }

def process_function_call(fn):
    if fn.name == 'entity':
        return find_entity(**json.loads(fn.arguments))
    else:
        return unknown_function(fn.name)

def fmt(str):
    formatted_lines = [textwrap.fill(line, width=120) for line in str.split('\n')]
    return '\n'.join(formatted_lines)


def chatbot(conversation, model="gpt-3.5-turbo-16k", temperature=0, functions_to_use = None):
    max_retry = 7
    retry = 0
    while True:
        try:
            if functions_to_use:
                response = openai.ChatCompletion.create(model=model, messages=conversation, functions=functions_to_use, temperature=temperature)
            else:
                response = openai.ChatCompletion.create(model=model, messages=conversation, temperature=temperature)

            if response['choices'][0]['finish_reason'] == 'function_call':
                conversation.append(process_function_call(response['choices'][0]['message']['function_call']))
            else:
                text = response['choices'][0]['message']['content']
                return text, response['usage']['total_tokens']
        except Exception as oops:
            print(f'\n\nError communicating with OpenAI: "{oops}"')
            if 'maximum context length' in str(oops):
                a = conversation.pop(0)
                print('\n\n DEBUG: Trimming oldest message')
                continue
            retry += 1
            if retry >= max_retry:
                print(f"\n\nExiting due to excessive errors in API: {oops}")
                exit(1)
            print(f'\n\nRetrying in {2 ** (retry - 1) * 5} seconds...')
            sleep(2 ** (retry - 1) * 5)


class GPTChatBot():

    def __init__(self):
        self._history = list()
        openai.api_key = open_file('key_openai.txt').strip()
        self._system_default = open_file('system_default.txt')
        self._user_profile = open_file('user_profile.txt')
        self._context = open_file('context.txt')
        self._update_user_profile = open_file('system_update_user_profile.txt')
        self._update_context = open_file('system_update_context.txt')
        self._system_message = self._system_default.replace('<<PROFILE>>', self._user_profile).replace('<<CONTEXT>>', self._context)

    def __call__(self, args):
        return self.generate_chat_response(self.compose_conversation(args))

    def compose_conversation(self, text):
        # continue with composing conversation and response
        self._history.append({'role': 'user', 'content': text})
        conversation = list()
        conversation += self._history
        self._system_message = self._system_default.replace('<<PROFILE>>', self._user_profile).replace('<<CONTEXT>>', self._context)
        conversation.append({'role': 'system', 'content': self._system_message})
        return conversation

    def generate_chat_response(self, conversation):
        # generate a response
        response, tokens = chatbot(conversation, functions_to_use=chatbot_functions)
        if tokens > 15800:
            z = self._history.pop(0)
        self._history.append({'role': 'assistant', 'content': response})
        return fmt(response)


    def history(self, _):
        formatted_lines = []
        for h in self._history:
            formatted_lines.append(f'Role: {h["role"]}')
            formatted_lines.append(f'Content: {h["content"]}')
        return '\n'.join(formatted_lines)
    
    def system_prompt(self, _):
        return self._system_message
    
    def user_profile(self, _):
        return self._user_profile
    
    def current_context(self, _):
        return self._context
    
    def update_user_profile(self, _):
        conversation = list()
        conversation += self._history
        update_profile = self._update_user_profile.replace('<<UPD>>', self._user_profile)
        conversation.append({'role': 'system', 'content': update_profile})
        response, _ = chatbot(conversation)
        self._user_profile = response
        return response

    def update_context(self, _):
        conversation = list()
        conversation += self._history
        update_context = self._update_context.replace('<<CONTEXT>>', self._context)
        conversation.append({'role': 'system', 'content': update_context})
        response, _ = chatbot(conversation)
        self._context = response
        return response

    
    # define a closure to allow class functions to be used in commands
    def make_command(self, func):
        def callable_entity(args):
            return fmt(func(args))
        return callable_entity


def _exit(_):
    exit(0)


if __name__ == '__main__':
    # instantiate chatbot, variables
    interp = CommandInterpreter()
    bot = GPTChatBot()
    interp.add_default_command(bot)
    interp.add_command('exit', _exit)
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
        
