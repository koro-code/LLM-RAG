import os
import chainlit as cl

# --- Ollama (LLM local)
from custom_ollama_llm import OllamaLLM

# Instancier l'LLM
ollama_llm = OllamaLLM(temperature=0.7)

@cl.on_chat_start
async def on_chat_start():
    """
    Message de bienvenue au démarrage du chat.
    """
    await cl.Message(
        content=(
            "Bienvenue dans cette interface !\n\n"
            "Posez librement vos questions et je vais y répondre via Ollama.\n"
        )
    ).send()

@cl.on_message
async def on_message(msg: cl.Message):
    """
    Cette fonction est appelée à chaque fois que l'utilisateur envoie un message dans le chat.
    Elle se contente de transmettre la question à Ollama et d'afficher la réponse.
    """
    user_text = msg.content or ""
    user_text = user_text.strip()

    if not user_text:
        await cl.Message(content="Message vide, merci de réessayer.").send()
        return

    # Appel au LLM local (Ollama)
    response_text = ollama_llm.generate(user_text)

    # On renvoie la réponse à l'utilisateur
    await cl.Message(content=response_text).send()
