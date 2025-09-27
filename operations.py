import numpy as np
import nltk

def softmax(logits):
    
    # 1. Ensures no overflow
    exps = np.exp(logits - np.max(logits))
    
    # 2. Sum the exponentiated scores and divide to get probabilities
    return exps / np.sum(exps)


def forwardpass(inp, weight_input, bias_inp, weight_output, bias_op):
    a = inp @ weight_input + bias_inp # matrix multiplication  [0.01, -0.03, 0.1 .....]
    activated_hidden = np.maximum(0, a) # ReLu funtion [0.01, 0, 0.1 .....]
    logits = activated_hidden @ weight_output + bias_op # [10, 15, 7, 5 .....]
    probabilities = softmax(logits) #[10, 15, 7, 5 .....] ----> [0.40, 0.32, 0.35 .....] sum ---> 1 or 0.999999
    return a, probabilities, activated_hidden


def preprocess_sentence(input_sentence, vocabulary, stemmer):
    tokenized_sentence = nltk.word_tokenize(input_sentence)

    stemmed_words = []
    for word in tokenized_sentence:
        stemmed_words.append(stemmer.stem(word.lower()))

    bag = np.zeros(len(vocabulary), dtype=np.float32)
    for idx, word in enumerate(vocabulary):
        if word in stemmed_words:
            bag[idx] = 1.0

    return bag
