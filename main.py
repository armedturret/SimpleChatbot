from chatbot_model import chatbot_model

cm = chatbot_model('tokenizer.pickle')
cm.load_from_file('checkpoints/cp.ckpt')

usr_input = ''
while usr_input != 'quit':
    print('Type something: ')
    usr_input = input()
    print(cm.predict(usr_input))