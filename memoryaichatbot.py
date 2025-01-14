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
