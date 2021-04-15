import PySimpleGUI as sg
import threading
import numpy as np
import io
import time
from PIL import Image

loading_layout = [
    [
        sg.Image("images/loading.png")
    ],
    [
        sg.Text("Loading Theodore...")
    ]
]

results = [None]

def load_model(window, results):
    from chatbot_model import chatbot_model
    cm = chatbot_model('tokenizer.pickle')
    cm.load_from_file('checkpoints/cp.ckpt')
    window.write_event_value('-THREAD DONE-', '')
    results[0] = cm

loading_window = sg.Window("Theodore V1.0", loading_layout, no_titlebar=True)
threading.Thread(target=load_model, args=(loading_window, results), daemon=True).start()
while True:
    event, values = loading_window.read()
    if event == '-THREAD DONE-':
        loading_window.close()
    elif event == sg.WIN_CLOSED:
        break

layout = [
    [
        sg.Listbox(size=(60, 20), values=[], key="-CHAT-"),
        sg.Image("images/neutral.png", key="-ICON-")
    ],
    [
        sg.Text("Please type your input here:")
    ], 
    [
        sg.In(size=(40, 1), enable_events=True, key="-INPUT-"),
        sg.Button("Send")
    ],
]

window = sg.Window("Theodore V1.0", layout)

def load_image(path):
    image = Image.open(path)
    image.thumbnail((256, 256))
    bio = io.BytesIO()
    image.save(bio, format="PNG")
    return bio.getvalue()

NEUTRAL = load_image("images/neutral.png")
THINKING = load_image("images/thinking.png")
SPEAKING = load_image("images/speaking.png")

# estimate of syllables in a word from https://stackoverflow.com/questions/46759492/syllable-count-in-python
def syllable_count(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count <= 0:
        count = 1
    return count

#calculates a delay based on complexity of a sentence using the equation https://aisel.aisnet.org/ecis2018_rp/113/
def calculate_delay(sentence):
    words = len(sentence.split(' '))
    syllables = 0
    for word in sentence.split(' '):
        syllables += syllable_count(word)
    C = 0.39 * words + 11.8 * (syllables / words) - 15.59
    if C < 0:
        C = 0
    return 0.5 * np.log(C + 0.5) + 1.5

chat = []
is_thinking = False
def wait_time(window, duration, event):
    time.sleep(duration)
    window.write_event_value(event, '')
    
while True:
    event, values = window.read()
    if event == "Send" and not is_thinking:
        #take in the input
        usr_input = values["-INPUT-"]
        chat.append("You: " + usr_input)
        window["-CHAT-"].update(chat)
        window["-INPUT-"].update("")
        #predict what to say
        prediction = results[0].predict(usr_input)
        delay = calculate_delay(prediction)
        chat.append("Theodore: " + prediction)
        window["-ICON-"].update(data=THINKING)
        threading.Thread(target=wait_time, args=(window, delay,"-THOUGHT-"), daemon=True).start()
        is_thinking = True
    elif event == "-THOUGHT-":
        #update the chat and reset the input
        window["-CHAT-"].update(chat)
        window["-ICON-"].update(data=SPEAKING)
        threading.Thread(target=wait_time, args=(window, delay,"-NORMAL-"), daemon=True).start()
        is_thinking = False
    elif event == "-NORMAL-" and not is_thinking:
        window["-ICON-"].update(data=NEUTRAL)
    elif event == "Exit" or event == sg.WIN_CLOSED:
        break