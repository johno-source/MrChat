#
# This module contains the capabilities used by the Input Cognition Node.
#

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

def search_web(search_terms):
    return "not implemented"