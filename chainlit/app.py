import os
from typing import List, Tuple
import uuid

import chainlit as cl
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM

from memory import PostgresConversationBufferMemory

import psycopg2
from psycopg2.extras import RealDictCursor

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_DB = os.getenv("POSTGRES_DB", "mydb")
POSTGRES_USER = os.getenv("POSTGRES_USER", "myuser")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "mypass")

print("[DEBUG] Attempting to connect to Postgres with:")
print(f"[DEBUG]   HOST={POSTGRES_HOST}, DB={POSTGRES_DB}, USER={POSTGRES_USER}, PASS={POSTGRES_PASSWORD}")

try:
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )
    print("[DEBUG] Connection to Postgres succeeded.")
except Exception as e:
    print("[ERROR] Could not connect to Postgres:", e)
    raise e


@cl.on_chat_start
async def on_chat_start():
    print("[DEBUG] on_chat_start triggered.")

    chain = cl.user_session.get("chain")
    if chain:
        print("[DEBUG] Found an existing chain in user_session. Not recreating it.")
        await cl.Message(content="Re-bonjour ! Posez votre question.").send()
        return

    print("[DEBUG] No existing chain found. Creating a new chain now.")

    session_id = cl.user_session.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        cl.user_session.set("session_id", session_id)
        print(f"[DEBUG] Created a new session_id = {session_id}")
    else:
        print(f"[DEBUG] Reusing existing session_id = {session_id}")

    # On demande un premier fichier avant de commencer la conversation
    files = None
    while files is None:
        print("[DEBUG] Asking user to upload a file...")
        files = await cl.AskFileMessage(
            content="Veuillez télécharger un fichier texte pour commencer !",
            accept=["text/plain"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]
    msg = cl.Message(content=f"Traitement de `{file.name}`...")
    await msg.send()
    print(f"[DEBUG] Received file: {file.name} at path: {file.path}")

    with open(file.path, "r", encoding="utf-8") as f:
        initial_text = f.read()

    # Split te text dans difents chunks pour l'indexation
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    initial_texts = text_splitter.split_text(initial_text)
    initial_metadatas = [{"source": f"init_{i}-pl"} for i in range(len(initial_texts))]

    embeddings = OpenAIEmbeddings()
    print("[DEBUG] Created embeddings with OpenAIEmbeddings.")

    # On crée le VectorStore ChromaBD
    docsearch = Chroma.from_texts(
        initial_texts,
        embeddings,
        metadatas=initial_metadatas,
        persist_directory="./chroma_db"
    )
    print("[DEBUG] Created initial Chroma vectorstore with persisted directory='./chroma_db'.")

    # On stocke les `embeddings` et les `text_splitter` pour réutiliser plus tard
    cl.user_session.set("embeddings", embeddings)
    cl.user_session.set("text_splitter", text_splitter)

    # Param du model qui sera utilisé 
    ollama_llm = OllamaLLM(
        base_url="http://ollama_app:11411",
        model="llama3:latest",
        temperature=0.7,
    )
    print("[DEBUG] Created OllamaLLM with model='llama3:latest'.")

    # On crée la mémoire de conversation via l'id de la session et la connexion Postgres
    memory = PostgresConversationBufferMemory(session_id=session_id, connection=conn)
    print(f"[DEBUG] Memory created: {memory}")

    # Chaîne RAG principale
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=ollama_llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )
    print("[DEBUG] ConversationalRetrievalChain created (rag_chain).")

    # On crée le prompt qui sera donnée au llm 
    fallback_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template=(
            "You are a helpful assistant.\n\n"
            "Here is the conversation so far:\n{chat_history}\n\n"
            "The user just asked: {question}\n"
            "Please answer in a conversational style, referencing earlier info if needed.\n"
        ),
    )
    fallback_chain = LLMChain(
        llm=ollama_llm,
        prompt=fallback_prompt,
        memory=memory,
    )
    print("[DEBUG] LLMChain created (fallback_chain).")

    # Stocker tout
    cl.user_session.set("chain", rag_chain)
    cl.user_session.set("fallback_chain", fallback_chain)
    cl.user_session.set("ollama_llm", ollama_llm)
    cl.user_session.set("docsearch", docsearch)
    print("[DEBUG] Stored rag_chain, fallback_chain, ollama_llm, docsearch in user_session.")

    msg.content = f"Traitement de `{file.name}` terminé. Vous pouvez maintenant poser vos questions ou uploader d'autres fichiers texte."
    await msg.update()
    print("[DEBUG] on_chat_start completed successfully.")


@cl.on_message
async def main(message: cl.Message):
    """
    1) Ingestion de fichier(s) texte (si l'utilisateur envoie en même temps un ou plusieurs .txt)
    2) Ensuite, on traite la question (fallback ou RAG), en s'appuyant sur les nouveaux documents.
    """
    print(f"[DEBUG] on_message triggered with user message: {message.content}")

    rag_chain: ConversationalRetrievalChain = cl.user_session.get("chain")
    fallback_chain: LLMChain = cl.user_session.get("fallback_chain")
    docsearch: Chroma = cl.user_session.get("docsearch")

    if not rag_chain or not docsearch:
        print("[WARNING] No chain/docsearch in user_session. Possibly user restarted or never uploaded file.")
        await cl.Message(content="Aucune chaîne initialisée. Veuillez redémarrer la conversation.").send()
        return

    # Vérifier s'il y a des fichiers dans 'message.elements'
    new_text_files = []
    if message.elements:
        embeddings: OpenAIEmbeddings = cl.user_session.get("embeddings")
        text_splitter: RecursiveCharacterTextSplitter = cl.user_session.get("text_splitter")

        # Filtrer uniquement .txt
        for e in message.elements:
            if e.name.lower().endswith(".txt"):
                new_text_files.append(e)

        # On ingère d'abord avant de traiter la question pour que la question puisse être basée sur les docs
        if new_text_files:
            info_msg = "Importing your new file(s) into the vectorstore:\n"
            for file_element in new_text_files:
                print(f"[DEBUG] Found new .txt file: {file_element.name}, path={file_element.path}")
                with open(file_element.path, "r", encoding="utf-8") as f:
                    content = f.read()

                splitted = text_splitter.split_text(content)
                metadatas = [{"source": f"uploaded_{file_element.name}_{i}"} for i in range(len(splitted))]
                # On ajoute au docsearch existant
                docsearch.add_texts(splitted, metadatas=metadatas)

                info_msg += f"- {file_element.name} ({len(splitted)} chunk(s))\n"

            await cl.Message(content=info_msg).send()

    # S'il n'y a pas de question textuelle on s'arrête
    user_input_text = message.content.strip()
    if not user_input_text:
        print("[DEBUG] No user text input (maybe just file upload).")
        return

    # Si il y a une question on la traite
    cb = cl.AsyncLangchainCallbackHandler()
    query = user_input_text

    # On vérifie si les docs sont pertinents pour repondre à la question 
    docs, below_threshold = get_relevant_docs_or_none(docsearch, query, threshold=0.3, k=3)
    print(f"[DEBUG] get_relevant_docs_or_none -> below_threshold={below_threshold}, docs={[d.page_content[:30] for d in docs]}")

    if below_threshold:
        print("[DEBUG] below_threshold=True => fallback_chain")
        res = await fallback_chain.ainvoke({"question": query}, callbacks=[cb])
        answer = res["text"]
        print("[DEBUG] fallback_chain answer:", answer)
        await cl.Message(content=answer).send()
        return

    print("[DEBUG] Doing RAG call (rag_chain.ainvoke).")
    res = await rag_chain.ainvoke({"question": query}, callbacks=[cb])

    answer = res["answer"]
    source_documents = res["source_documents"]
    print("[DEBUG] LLM answer:", answer)

    # Ajouter les sources en side-bar
    text_elements = []
    if source_documents:
        for i, source_doc in enumerate(source_documents):
            name = f"source_{i}"
            text_elements.append(cl.Text(content=source_doc.page_content, name=name, display="side"))
        source_names = [te.name for te in text_elements]
        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"

    await cl.Message(content=answer, elements=text_elements).send()
    print("[DEBUG] on_message completed.")


def get_relevant_docs_or_none(docsearch: Chroma, query: str, threshold=0.3, k=3) -> Tuple[List[Document], bool]:
    print(f"[DEBUG] get_relevant_docs_or_none called with query='{query}', threshold={threshold}, k={k}")
    if not hasattr(docsearch, "similarity_search_with_score"):
        print("[DEBUG] docsearch has no similarity_search_with_score, using docsearch.similarity_search...")
        docs = docsearch.similarity_search(query, k=k)
        if not docs:
            return [], True
        return docs, False

    # On utilise similarity_search_with_score pour avoir les scores de similarité pour savoir si les docs peuvent etre utilisés
    docs_with_scores = docsearch.similarity_search_with_score(query, k=k)
    print("[DEBUG] similarity_search_with_score returned", docs_with_scores)
    if not docs_with_scores:
        return [], True

    relevant_docs = []
    all_below = True
    for doc, score in docs_with_scores:
        print(f"[DEBUG] doc='{doc.page_content[:30]}', score={score}")
        if score < threshold:
            all_below = False
        relevant_docs.append(doc)

    if all_below:
        return [], True
    else:
        return relevant_docs, False
