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

    # session_id
    session_id = cl.user_session.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        cl.user_session.set("session_id", session_id)
        print(f"[DEBUG] Created a new session_id = {session_id}")
    else:
        print(f"[DEBUG] Reusing existing session_id = {session_id}")

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
        text = f.read()

    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(text)
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Embeddings
    embeddings = OpenAIEmbeddings()
    print("[DEBUG] Created embeddings with OpenAIEmbeddings.")

    # VectorStore
    docsearch = Chroma.from_texts(
        texts,
        embeddings,
        metadatas=metadatas,
        persist_directory="./chroma_db"
    )
    print("[DEBUG] Created Chroma vectorstore with persisted directory='./chroma_db'.")

    # LLM
    ollama_llm = OllamaLLM(
        base_url="http://ollama_app:11411",
        model="llama3:latest",
        temperature=0.7,
    )
    print("[DEBUG] Created OllamaLLM with model='llama3:latest'.")

    # Mémoire Postgres
    memory = PostgresConversationBufferMemory(session_id=session_id, connection=conn)
    print(f"[DEBUG] Memory created: {memory}")

    # Chaîne RAG principale
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=ollama_llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
        # verbose=True
    )
    print("[DEBUG] ConversationalRetrievalChain created (rag_chain).")

    # CHANGEMENT 2 : Prompt qui inclut chat_history pour le fallback
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
        memory=memory,   # <= important pour que chat_history soit chargé
    )
    print("[DEBUG] LLMChain created (fallback_chain).")

    # Stocker tout
    cl.user_session.set("chain", rag_chain)
    cl.user_session.set("fallback_chain", fallback_chain)
    cl.user_session.set("ollama_llm", ollama_llm)
    cl.user_session.set("docsearch", docsearch)
    print("[DEBUG] Stored rag_chain, fallback_chain, ollama_llm, docsearch in user_session.")

    msg.content = f"Traitement de `{file.name}` terminé. Posez vos questions !"
    await msg.update()
    print("[DEBUG] on_chat_start completed successfully.")


@cl.on_message
async def main(message: cl.Message):
    print(f"[DEBUG] on_message triggered with user message: {message.content}")

    rag_chain: ConversationalRetrievalChain = cl.user_session.get("chain")
    fallback_chain: LLMChain = cl.user_session.get("fallback_chain")
    docsearch: Chroma = cl.user_session.get("docsearch")

    if not rag_chain:
        print("[WARNING] No chain in user_session. Possibly user restarted or never uploaded file.")
        await cl.Message(
            content="Aucune chaîne initialisée. Veuillez redémarrer la conversation."
        ).send()
        return

    cb = cl.AsyncLangchainCallbackHandler()
    query = message.content

    docs, below_threshold = get_relevant_docs_or_none(docsearch, query, threshold=0.3, k=3)
    print(f"[DEBUG] get_relevant_docs_or_none -> below_threshold={below_threshold}, docs={[d.page_content[:30] for d in docs]}")

    if below_threshold:
        print("[DEBUG] below_threshold = True => fallback_chain.")
        # On invoque la fallback chain => TOUT sera mémorisé, y compris la réponse
        # comme on a memory=..., ça injectera chat_history et question
        res = await fallback_chain.ainvoke({"question": query}, callbacks=[cb])
        # res est un dict => {"text": "some answer"}
        answer = res["text"]
        print("[DEBUG] fallback_chain answer:", answer)
        await cl.Message(content=answer).send()
        return

    print("[DEBUG] Doing RAG call (rag_chain.ainvoke).")
    res = await rag_chain.ainvoke({"question": query}, callbacks=[cb])

    answer = res["answer"]
    source_documents = res["source_documents"]
    print("[DEBUG] LLM answer:", answer)

    text_elements = []
    if source_documents:
        for i, source_doc in enumerate(source_documents):
            name = f"source_{i}"
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=name, display="side")
            )
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

    docs_with_scores = docsearch.similarity_search_with_score(query, k=k)
    print("[DEBUG] similarity_search_with_score returned", docs_with_scores)
    if not docs_with_scores:
        return [], True

    relevant_docs = []
    all_below = True
    for doc, score in docs_with_scores:
        print(f"[DEBUG] doc='{doc.page_content[:30]}', score={score}")
        if score < threshold:
            # On détecte qu'au moins un doc est sous le threshold
            all_below = False
        relevant_docs.append(doc)

    if all_below:
        return [], True
    else:
        return relevant_docs, False
