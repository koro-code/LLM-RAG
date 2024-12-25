import chainlit as cl
from langchain.chains import ConversationChain
from langchain.llms import Ollama

# Initialisation du LLM Ollama
llm = Ollama(
    model="llama3:latest"  # Remplacez par le nom de votre modèle Ollama
)

# Configuration de la chaîne de conversation LangChain
conversation = ConversationChain(
    llm=llm
)

@cl.on_message
async def main(message: cl.Message):
    # Logique pour interagir avec le modèle LLM via LangChain
    user_message = message.content
    response = conversation.run(user_message)

    # Envoie la réponse du LLM à l'utilisateur
    await cl.Message(
        content=f"LLM Response: {response}",
    ).send()
