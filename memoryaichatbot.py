# This chatbot uses memory
pip install transformers
from transformers import AutoModelForCausalLM, AutoTokenizer # LM means language modeling
import torch # Is a library for working with tensors (multi-dimensional arrays) and performing operations on them
# AI models like DialoGPT or GPT-3 work w/numbers because they perform mathematical operations like matrix multiplications to understand and generate text
# Text is converted into numbers called TOKEN IDs for the model to actyalky
