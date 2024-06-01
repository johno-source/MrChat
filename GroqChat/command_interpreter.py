#
# This is a command interpreter that takes a string and works out what command to execute in response
# It is a very simple version that assumes the command is uniquely identified in the first word.
# the remainder of the command is interpreted as arguments to the command.
#
# This class acts like a singleton in that it shares its data across all instances.
from concurrent.futures import ThreadPoolExecutor

class CommandInterpreter():

    # all the commands are kept in a dictionary
    _commands = {}
    _default_command = None
    _executor = ThreadPoolExecutor(max_workers=1)

    def __init__(self):
        if not self._default_command:
            self._default_command = self.default_command


    def add_command(self, signature, executable):
        # only use the first word in the signature
        words = signature.split()

        # no checking for duplicates
        self._commands[words[0]] = executable

    def add_default_command(self, executable):
        self._default_command = executable


    # this is the heart of the interpreter - find and execute the command.
    def execute(self, command, args = None):
        if command in self._commands:
            print(f"Executing command {command} with args {args}")
            return self._executor.submit(self._commands[command], args).result()
        else:
            print(f"Executing default command {command}")
            return self._executor.submit(self._default_command, command).result()
        
    def default_command(self, command):
        words = command.split()
        return f"Unknown command {words[0]} in {command}"
