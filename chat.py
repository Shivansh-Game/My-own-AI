import random
import json
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
from operations import preprocess_sentence, forwardpass

data = np.load("trained_model.npz", allow_pickle=True)
weight_input = data['W1']
bias_inp = data['b1']
weight_output = data['W2']
bias_op = data['b2']
vocabulary = data['vocabulary']
tags = data['tags']

with open('intents.json', 'r') as f:
    intents = json.load(f)

stemmer = PorterStemmer()

print("Bot is ready! Type 'quit' to exit.")
while True:
    user_sentence = input("You: ")
    if user_sentence.lower() == "quit":
        break

    bow_vector = preprocess_sentence(user_sentence, vocabulary, stemmer)
    
    probabilities = forwardpass(bow_vector, weight_input, bias_inp, weight_output, bias_op)[1]

    max_prob = np.max(probabilities)
    predicted_tag_index = np.argmax(probabilities)
    
    response = ""
    print(max_prob)
    if max_prob > 0:
        predicted_tag = tags[predicted_tag_index]
        
        for intent in intents['intents']:
            if intent['tag'] == predicted_tag:
                response = random.choice(intent['responses'])
                break
    else:
        for intent in intents['intents']:
            if intent['tag'] == 'fallback':
                response = random.choice(intent['responses'])
                break

    print(f"Bot: {response}")
