#
# utilities used by scalzi
#
import textwrap
from langchain_core.messages.ai import AIMessage

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
        return infile.read()

def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def append_file(filepath, content):
    with open(filepath, 'a', encoding='utf-8') as outfile:
        outfile.write(content)


results_log_file = 'logs/log.txt'
def log_results(results):
    append_file(results_log_file, results + '\n')

def fmt(str):
    formatted_lines = [textwrap.fill(line, width=120) for line in str.split('\n')]
    return '\n'.join(formatted_lines)

def FilterOutExtraToJSON(input):
    candidate_json = input.content
    posi = candidate_json.find('{')
    return AIMessage(content=candidate_json[posi:])

def remove_quotes(s):
    return s.lstrip(' "\'').rstrip(' "\'')


