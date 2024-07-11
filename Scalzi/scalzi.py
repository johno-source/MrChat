#
#                                                                           Synopsis
# Based on a YouTube video (https://www.youtube.com/watch?v=pi6gr_YHSuc&list=PLIi9WZDjBK2SEUyLj5t3Nj5MA3gFYhrig) where the presenter created a chatbot 
# called Jarvis, I was inspired to copy it. I found that his function selection did not work as well as I would like so I refined it and am pleased to
# the point that I am now going to create a bot based on his design. I have further enhanced it by adding the concept of a Chat Ledger, which will become the 
# basis for memory creation and recall.
# He has also convinced me to abandon the use of Ollama and move to using groq. It adds some limits but they reset every day and I doubt that I can reach
# them anyway. It is a good choice, it is fast, and it gives me access to Llama 3 70b which I cannot run on any GPU that I could buy. (Well I could buy a
# Mac Studio but a suitable one will cost around 10K. Until groq came along I was considering it and still might.)
# I have chosen the name scalzi for my chatbot. It is the name of the author I am currently reading. I wanted a name that was short but distinctive in case I 
# add a voice interface.
# The further I go, however, the more my ideas have evolved and the design with it. I want to introduce the notion of a cognition node. A cognition node is
# comprised of an LLM that is prompted to return data in JSON format. I have found that prompting it to give a thought as the first field first lets you see
# its reasoning for the decisions it has made but secondly it causes the LLM to think through step by step, giving better answers. (I have not measured this.
# But it does feel like the answers are better.) I plan to build on this by having the AI beign able to explain all of its decisions. The JSON output may select
# a function to be called that adds further information for subsequent steps or selects a new chain of cognition nodes to be invoked. Each cognition node takes
# a chat ledger as input. The node is free to update the ledger as it goes. The other ways a node can give output is by directing output to the user or by
# generating a memory. At first I thought I would simply store the ledger at a point in time as a memory, but thinking about it I am going to let the memory
# store node use the chat ledger to create one or more memories. I suspect that the finer grain a memory is the better but it needs to have the ability to find
# other memories by association. 
# As I expect there to be many memories being recalled and we only have a limited context window (8K on Llama 3 70b) I am going to use Sparse Primer Representations
# to store and retrieve the data in compressed format. This does add an extra LLM step when forming memories but I hope to do it in the background so the user
# will not notice. 
# I found the voice interface from the original video (above) to be fairly clunky. It only complicates things so at this stage I am sticking to a text interface
# and may move to Chainlit for a web interface in the future.
#

from command_interpreter import CommandInterpreter
from command_interpreter import make_command
from context_cognition_node import ContextCognitionNode
from sensory_cognition_node import SensoryCognitionNode
from chat_cognition_node import ChatCognitionNode
from chat_ledger import ChatLedger
from langchain_groq import ChatGroq
from scalzi_logger import *

class ThoughtChain:
    def __init__(self, interp):
        # this holds the chatbots state
        self.ledger = ChatLedger()
        self.context_node = ContextCognitionNode()
        self.sensory_node = SensoryCognitionNode()
        self.chat_node = ChatCognitionNode()

        # set up the commands to access the contexts
        interp.add_command('/context', make_command(self.context_command))
        interp.add_command('/sensory', make_command(self.sensory_command))
        interp.add_command('/chat', make_command(self.chat_command))

    # this is the input point for the chat bot - everything is orchestrated from here
    def __call__(self, args):
        self.ledger.set_user_input(args)
        self.context_node.run(self.ledger)
        self.sensory_node.run(self.ledger)
        self.chat_node.run(self.ledger)
        return self.ledger.get_chat_output()

    # define some functions to access the ledger
    def get_user_input(self, _):
        return self.ledger.get_user_input()

    def set_user_input(self, args):
        self.ledger.set_user_input(args)
        return f'User input set to: {args}'

    # permit each node to be invoked from the command line
    def context_command(self, _):
        return self.context_node.run(self.ledger)

    def sensory_command(self, _):
        return self.sensory_node.run(self.ledger)

    def chat_command(self, _):
        return self.chat_node.run(self.ledger)

    def exit(self,_):
        print("Bye!")
        exit(0)

def set_file_log_level_command(level):
    return f'File logging level set to: {set_file_log_level(level)}'

def set_console_log_level_command(level):
    return f'Console logging level set to: {set_console_log_level(level)}'

def log_command(args):
    words = args.split()
    if len(words) == 2:
        if words[0] == 'file':
            return set_file_log_level_command(words[1])
        if words[0] == 'console':
            return set_console_log_level_command(words[1])
    return 'Usage: log <file|console> <level>'

if __name__ == '__main__':
    # The user interface is all run through a command interpreter
    interp = CommandInterpreter()
    thought_chain = ThoughtChain(interp)
    interp.add_default_command(thought_chain)
    interp.add_command('exit', make_command(thought_chain.exit))
    interp.add_command('log', make_command(log_command))
    interp.add_command('get_user', make_command(thought_chain.get_user_input))
    interp.add_command('set_user', make_command(thought_chain.set_user_input))

    while True:
        text = input('\nScalzi> ')
        if text:
            print(interp.execute(text))
        
