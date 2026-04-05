# NumPy Chatbot - A Neural Network from Scratch

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3.8-yellow.svg)

A fully functional, intent-based chatbot built from the ground up using only **NumPy**. This project demonstrates a fundamental understanding of neural networks by implementing the entire training process, including **backpropagation**, without the use of high-level deep learning frameworks like PyTorch or TensorFlow.

## About The Project

This project is a command-line chatbot created to showcase the core mechanics of Natural Language Processing and Deep Learning. The bot can understand user intent based on a predefined dataset (`intents.json`) and provide a relevant, randomized response.

The key feature of this project is that the feedforward neural network that powers the bot is written entirely from scratch. This includes data preprocessing, the forward pass, cross-entropy loss calculation, and the complete backpropagation algorithm for training.

## Features

The chatbot is trained to recognize and respond to a variety of intents, including:
* Greetings and Goodbyes
* Jokes and expressions of gratitude
* Questions about the bot's creator and purpose
* Handling of simple insults and praise
* Recognizing user emotions (happy/sad)

## How It Works

1.  **Data Preprocessing:** The text from `intents.json` is tokenized, stemmed, and converted into a vocabulary. Each sentence is then transformed into a **Bag-of-Words** vector.
2.  **Neural Network Architecture:** The model is a simple feedforward neural network with one hidden layer:
    * **Input Layer:** Size of the vocabulary.
    * **Hidden Layer:** 64 neurons with a **ReLU** activation function.
    * **Output Layer:** Number of intents with a **Softmax** activation function to produce probabilities.
3.  **Training:** The network is trained for 5000 epochs using **Stochastic Gradient Descent (SGD)** and a custom-coded **backpropagation** algorithm to minimize the cross-entropy loss.
4.  **Inference:** The live chat script (`chat.py`) takes a user's sentence, converts it into a Bag-of-Words vector, and performs a forward pass through the trained network to get the intent probabilities and select the best response.

## Tech Stack

* [Python](https://www.python.org/)
* [NumPy](https://numpy.org/) - For all numerical operations and the neural network implementation.
* [NLTK](https://www.nltk.org/) - For natural language processing tasks like tokenization and stemming.

## Getting Started

Follow these instructions to get a local copy up and running.

### Prerequisites

* Python 3.9+
* Git

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/Shivansh-Game/My-own-AI.git](https://github.com/Shivansh-Game/My-own-AI.git)
    cd My-own-AI
    ```
2.  **Create and activate a virtual environment:**
    ```sh
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```
4.  **Download NLTK data:**
    * Run Python in your terminal: `python`
    * In the Python interpreter, run the following:
        ```python
        import nltk
        nltk.download('punkt')
        exit()
        ```
5.  **Train the model:**
    ```sh
    python train.py
    ```
    This will create a `trained_model.npz` file containing the trained weights and biases.

6.  **Run the chatbot:**
    ```sh
    python chat.py
    ```
    You can now chat with the bot in your terminal! Type `quit` to exit.

