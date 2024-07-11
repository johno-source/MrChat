#
# This module contains the capabilities used by the Input Cognition Node.
#
from langchain_community.tools.tavily_search import TavilySearchResults
from scalzi_logger import scalzi_logger


def take_screenshot():
    path = 'screenshot.jpg'
    screenshot = ImageGrab.grab()
    rgb_screenshot = screenshot.convert('RGB')
    rgb_screenshot.save(path, quality=15)


def capture_webcam():
    if not web_cam.isOpened():
        print('Error: Camera did not open successfully')
        return
    
    path = 'webcam.jpg'
    ret, frame = web_cam.read()
    cv2.imwrite(path, frame)


def read_clipboard():
    clipboard_content = pyperclip.paste()
    if isinstance(clipboard_content, str):
        return clipboard_content
    else:
        print('Error: Clipboard content is not a string')
        return "None"

_search = TavilySearchResults()
def search_web(search_terms):
    # the search terms might be a string or a list
    if isinstance(search_terms, str):
        search_str = search_terms
    else:
        search_str = ' OR '.join(search_terms)

    # this returns a list of dictionaries 
    scalzi_logger.info(f'Searching web for: {type(search_str)} {search_str}')
    return search_str, _search.invoke({'query': search_str})[:3]