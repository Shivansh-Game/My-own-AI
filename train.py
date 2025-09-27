import numpy as np
import json
import nltk
from nltk.stem.porter import PorterStemmer
from operations import forwardpass

#This is a one-time download
#nltk.download('punkt')
#nltk.download('punkt_tab')

stemmer = PorterStemmer()

with open('intents.json', 'r') as f:
    intents = json.load(f)

print("Data loaded successfully!")


all_words = []
tags = []
xy = []


for intent in intents['intents']:
    tag = intent['tag']

    if tag not in tags:
        tags.append(tag)
        
    for pattern in intent['patterns']:
        
        words = nltk.word_tokenize(pattern)

        all_words.extend(words)

        xy.append((words, tag))


ignore_words = ['?', '.', '!']


all_words = [stemmer.stem(w.lower()) for w in all_words if w not in ignore_words]

vocabulary = sorted(list(set(all_words)))
tags = sorted(list(set(tags)))


X_train = []
y_train = []

for (pattern_sentence, tag) in xy:

    bag = []

    pattern_words = [stemmer.stem(w.lower()) for w in pattern_sentence]
    

    for word in vocabulary:
        bag.append(1) if word in pattern_words else bag.append(0)
        
    X_train.append(bag)
    

    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"Created training data. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

hidden_layer_size = 64

# full forward pass
weight_input = np.array([[np.random.randn() * 0.01 for _ in range(hidden_layer_size)] for _ in range(len(vocabulary))]) # 58, 8

weight_output = np.array([[np.random.randn() * 0.01 for _ in range(len(tags))] for _ in range(hidden_layer_size)]) # 8, 10

bias_inp = np.zeros(hidden_layer_size,)

bias_op = np.zeros(len(tags),)

# Done to give the AI diverse examples
num_examples = len(X_train)
indices = list(range(num_examples))


epochs = 5000
for epoch in range(epochs):
    # shuffle the indices to avoid overfitting
    np.random.shuffle(indices)
    for i in indices:
        inp = X_train[i]
        a, probabilities, activated_hidden = forwardpass(inp, weight_input, bias_inp, weight_output, bias_op)

        # loss calculation 

        loss = -np.log(probabilities[y_train[i]]) 

        # ----- Backpropagation ----- #

        # True answer vector 

        tav = np.zeros(len(tags),)
        tav[y_train[i]] = 1

        # initial error gradient

        ieg = probabilities - tav

        # ieg is also the gradient for bias_op

        grad_weight_output = np.outer(activated_hidden, ieg)

        hle = np.dot(ieg, weight_output.T)

        Relu_derivative = (a > 0).astype(int)

        hle = hle * Relu_derivative

        # hle is also the gradient for bias_inp

        grad_weight_input = np.outer(inp, hle)


        # define the learning rate

        LR = 0.001

        weight_output -= LR * grad_weight_output

        bias_op -= LR * ieg # ieg is also the gradient for bias_op

        weight_input -= LR * grad_weight_input

        bias_inp -= LR * hle # hle is also the gradient for bias_inp

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}')
            
            
print("Training complete.")

file_path = "trained_model.npz"

# Save post training weights and biases
np.savez("trained_model.npz", 
         W1=weight_input, 
         b1=bias_inp, 
         W2=weight_output, 
         b2=bias_op,
         vocabulary=vocabulary,
         tags=tags)

print(f"Model parameters saved to {file_path}")
