responses = { # Here, we are defining a dictionary of questions and responses
    "hello": "Hi there! How can I help you?",
    "how are you": "I'm just a bot, but I'm functioning as expected!",
    "bye": "Goodbye! Have a great day!",
}
def chatbot(input_text): # We are using a function called chatbot to handle the input from the user
    input_text = input_text.lower()
    return responses.get(input_text, "Sorry, I don't understand that.")
