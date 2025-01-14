pip install transformers
from transformers import pipeline # The pipeline utility is a high-level interface (a point of interaction between systems or components), designed to simplify process of working with pre-trained models, allows you to quickly load and use models for common tasks like text generation and classification, translation, and conversational AI, without actually needing to understand the details of the model or its architecture

chatbot = pipeline("text-generation", model="microsoft/DialoGPT-large")

while True:
  user_input = input("You: ")
  if user_input.lower() == "exit":
    print("Existing chatbot, goodbye.")
    break
  # The num_return_sequences parameter is a crucial aspect of text generation models, particularly in the context of generating multiple outputs from a single input prompt
  # This parameter allows users to specify how many different sequences the model should generate for a given input
  response = chatbot(user_input, max_length=50, num_return_sequences=1) # Number of responses/sequences the model should generate for a given input
  print("Bot:", response[0]["generated_text"])
  """
  response[0]
  What it is: Accesses the first element in the response list.
  Why it's needed: If num_return_sequences > 1 is used, multiple responses are generated and stored as a list. response[0] selects the first response.

  ["generated_text"]
  What it is: Accesses the "generated_text" key in the first response.
  Why it's needed: The dictionary returned for each response includes various keys, such as:
  "generated_text": The actual text generated by the model.
  Other keys might include metadata like token IDs or probabilities.
  """
