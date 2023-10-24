import random
import json
import torch

from model import NeuralNet
from function import contains_promotion
from nltk_utils import bag_of_words, tokenize

# Determine whether to use a GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents from a JSON file
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Load pre-trained data from a file
FILE = "data.pth"
data = torch.load(FILE)

# Extract essential data from the loaded file
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Initialize a neural network model and load pre-trained model weights
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Set the chatbot's name
bot_name = "Foodie"

# Start a conversation loop
print("Let's chat! (type 'quit' to exit)")
while True:
    # Get user input
    sentence = input("You: ")

    # Exit the loop if the user wants to quit
    if sentence == "quit":
        break

    # Check for promotional messages and provide a response if found
    if contains_promotion(sentence):
        print("Promotion available in koththu")
        continue

    # Tokenize the user's input and convert it to a bag of words
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    # Make predictions with the model
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Determine the intent and respond accordingly
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")
