# import os
# import chainlit as cl

# # Qdrant
# from qdrant_client import QdrantClient
# from qdrant_client.models import Distance, VectorParams

# # LangChain
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import Qdrant
# from langchain.embeddings.openai import OpenAIEmbeddings

# # Ollama custom
# from custom_ollama_llm import OllamaLLM

# QDRANT_HOST = os.getenv("VECTOR_DB_HOST", "qdrant")
# QDRANT_PORT = os.getenv("VECTOR_DB_PORT", "6333")
# collection_name = "my_collection"

# # Connexion Qdrant
# qdrant_client = QdrantClient(url=f"http://{QDRANT_HOST}:{QDRANT_PORT}")

# try:
#     qdrant_client.recreate_collection(
#         collection_name=collection_name,
#         vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
#     )
#     print(f"Collection '{collection_name}' recréée avec succès.")
# except Exception as e:
#     print("Erreur Qdrant:", e)

# # Embeddings (OpenAI, donc nécessite OPENAI_API_KEY)
# embedding_function = OpenAIEmbeddings()

# # LLM local (Ollama)
# ollama_llm = OllamaLLM(temperature=0.7)

# # Variable globale : retriever
# retriever = None

# def index_file_content(text_content: str):
#     """
#     Split, embedding, index into Qdrant.
#     Met à jour la variable globale `retriever`.
#     """
#     global retriever
#     splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     docs = splitter.create_documents([text_content])

#     vectorstore = Qdrant.from_documents(
#         docs,
#         embedding_function,
#         url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
#         collection_name=collection_name
#     )
#     retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})


# @cl.on_chat_start
# async def on_chat_start():
#     """Message de bienvenue."""
#     await cl.Message(
#         content=(
#             "Bienvenue !\n\n"
#             "- **Envoyez un message** pour poser une question.\n"
#             "- **Attachez un fichier** (en cliquant sur l'icône 'Attach') pour l'indexer."
#         )
#     ).send()


# @cl.on_message
# async def on_message(msg: cl.Message):
#     """
#     Gère à la fois l'upload (si des fichiers sont attachés)
#     et la question (si c'est seulement du texte).
#     """
#     user_text = (msg.content or "").strip()
#     user_files = msg.files or []  # liste de chainlit.UploadedFile

#     # 1) Si l'utilisateur a attaché un ou plusieurs fichiers
#     if len(user_files) > 0:
#         # On indexe chaque fichier
#         for f in user_files:
#             file_name = f.name
#             file_bytes = f.content  # bytes

#             # Pour un PDF, parsez vraiment le PDF (p. ex. PyPDF2).
#             # Ici on suppose un .txt ou .md qu'on peut .decode("utf-8")
#             try:
#                 file_text = file_bytes.decode("utf-8", errors="ignore")
#             except Exception as e:
#                 await cl.Message(content=f"Impossible de lire {file_name} : {e}").send()
#                 continue

#             index_file_content(file_text)

#         await cl.Message(content="Fichier(s) indexé(s) avec succès.").send()
#         return

#     # 2) Sinon, c'est une question
#     if not user_text:
#         await cl.Message(content="Message vide, merci de réessayer.").send()
#         return

#     # Vérifie si on a déjà un retriever
#     global retriever
#     if retriever is None:
#         await cl.Message(
#             content="Aucun document n'est encore indexé. Uploadez un fichier pour commencer."
#         ).send()
#         return

#     # Récupération des passages pertinents
#     related_docs = retriever.get_relevant_documents(user_text)
#     context = "\n\n".join(doc.page_content for doc in related_docs)

#     prompt = f"""
# Voici du contexte issu de votre base documentaire :
# {context}

# Question : {user_text}

# Réponds au mieux en citant le contenu si nécessaire.
# """
#     response_text = ollama_llm.generate(prompt)
#     await cl.Message(content=response_text).send()


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
