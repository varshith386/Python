import openai

def chat_with_gpt(api_key, max_turns=3):
    """
    Function to interact with ChatGPT-4 turbo for a limited number of turns. 
    After the conversation, it returns the last answer of the user.

    Parameters:
    api_key (str): Your OpenAI API key.
    max_turns (int): Maximum number of conversation turns.

    Returns:
    str: The last answer of the user.
    """

    openai.api_key = api_key
    conversation_history = []

    for turn in range(max_turns):
        user_input = input("You: ")
        conversation_history.append({"role": "user", "content": user_input})

        try:
            if turn == max_turns - 1:  # Insert the specific question on the last turn
                conversation_history.append({
                    "role": "system",
                    "content": "How is your life going?"
                })

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-1106",
                messages=conversation_history
            )
            bot_message = response.choices[0].message['content']
            print("Bot:", bot_message)
            conversation_history.append({"role": "system", "content": bot_message})

        except Exception as e:
            print("Error:", e)
            break

    # Return the last answer of the user
    return next((msg["content"] for msg in reversed(conversation_history) if msg["role"] == "user"), None)

# Example usage
api_key = "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"  # Replace with your actual API key
last_user_input = chat_with_gpt(api_key)

