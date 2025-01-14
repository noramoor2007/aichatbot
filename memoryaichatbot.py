# This chatbot uses memory
pip install transformers
from transformers import AutoModelForCausalLM, AutoTokenizer # LM means language modeling
import torch # Is a library for working with tensors (multi-dimensional arrays) and performing operations on them
# AI models like DialoGPT or GPT-3 work w/numbers because they perform mathematical operations like matrix multiplications to understand and generate text
# Text is converted into numbers called TOKEN IDs for the model to actyalky

# Load the DialoGPT model and tokenizer
model_name = "microsoft/DialoGPT-large" # This is the pre-trained model to use
tokenizer = AutoTokenizer.from_pretrained(model_name) # AutoTokenizer tokenizes the input text and converts it into numerical format for the model
model = AutoModelForCausalLM.from_pretrained(model_name) # AutoModelForCasualLM basically loads a pre-trained model for casual language modeling (used for text generation)
# Initialize the conversation history
conversation_history = None

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Exiting chatbot. Goodbye!")
        break

    # Encode the new user input and add it to the conversation history
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

    # Add the new input to the existing conversation history
    if conversation_history is not None and len(conversation_history) > 0:
      bot_input_ids = torch.cat([conversation_history, new_input_ids], dim=-1)
    else:
      bot_input_ids = new_input_ids
