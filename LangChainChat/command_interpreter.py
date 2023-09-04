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
    _help = {}
    _default_command = None
    _executor = ThreadPoolExecutor(max_workers=1)

    def __init__(self):
        if not self._default_command:
            self._default_command = self.default_command


    def add_command(self, signature, executable, command_help):
        # only use the first word in the signature
        words = signature.split()

        # no checking for duplicates
        self._commands[words[0]] = executable

        self._help[words[0]] = command_help

    def add_default_command(self, executable):
        self._default_command = executable


    # this is the heart of the interpreter - find and execute the command.
    def execute(self, command):
        words = command.split()
        cmd = words[0]
        if cmd in self._commands:
            return self._executor.submit(self._commands[cmd], words[1:]).result()
        else:
            return self._executor.submit(self._default_command, command).result()
        
    def default_command(self, command):
        words = command.split()
        return f"Unknown command {words[0]} in {command}"

    def help(self):
        formatted_lines = []
        for c, text in self._help.items():
            formatted_lines.append(f'{c:<20} : {text}')
        return '\n'.join(formatted_lines)

    def function_calls(self):
        formatted_lines = ["["]
        for c, text in self._help.items():
            formatted_lines.append( '    {')
            formatted_lines.append(f'        "name": "{c}",')
            formatted_lines.append(f'        "description": "{text}"')
            formatted_lines.append( '        "parameters": {')
            formatted_lines.append( '            "type": "object",')
            formatted_lines.append( '            "properties": {},')
            formatted_lines.append( '    },')
        formatted_lines.append("]")
        return '\n'.join(formatted_lines)


