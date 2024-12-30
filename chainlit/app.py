import os
from typing import List, Tuple

import chainlit as cl
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_ollama.llms import OllamaLLM


@cl.on_chat_start
async def on_chat_start():
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Veuillez télécharger un fichier texte pour commencer !",
            accept=["text/plain"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]
    msg = cl.Message(content=f"Traitement de `{file.name}`...")
    await msg.send()

    with open(file.path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(text)
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Embeddings
    embeddings = OpenAIEmbeddings()

    # Créer la base Chroma
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts,
        embeddings,
        metadatas=metadatas,
        persist_directory="./chroma_db"
    )

    # Mémoire
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # LLM Ollama
    ollama_llm = OllamaLLM(
        base_url="http://ollama_app:11411",
        model="llama3:latest",
        temperature=0.7,
    )

    # Chaîne RAG
    chain = ConversationalRetrievalChain.from_llm(
        llm=ollama_llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    msg.content = f"Traitement de `{file.name}` terminé. Posez vos questions !"
    await msg.update()

    cl.user_session.set("chain", chain)
    cl.user_session.set("ollama_llm", ollama_llm)
    cl.user_session.set("docsearch", docsearch)


@cl.on_message
async def main(message: cl.Message):
    chain: ConversationalRetrievalChain = cl.user_session.get("chain")
    ollama_llm: OllamaLLM = cl.user_session.get("ollama_llm")
    docsearch: Chroma = cl.user_session.get("docsearch")

    cb = cl.AsyncLangchainCallbackHandler()

    query = message.content

    # On essaye d'obtenir la similarité + documents
    docs, below_threshold = get_relevant_docs_or_none(docsearch, query, threshold=0.3, k=3)

    if below_threshold:
        generic_answer = _ollama_direct(ollama_llm, query)
        await cl.Message(content=generic_answer).send()
        return

    # Sinon, on fait un RAG normal
    res = await chain.ainvoke(query, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]

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


def get_relevant_docs_or_none(docsearch: Chroma, query: str, threshold=0.3, k=3) -> Tuple[List[Document], bool]:
    """
    1) Tente de récupérer jusqu'à k documents avec score.
    2) Si tous les scores sont inférieurs à `threshold`, on renvoie below_threshold=True
    3) Sinon, on renvoie la liste (même s'il y en a un tout petit peu pertinent)
    """
    if not hasattr(docsearch, "similarity_search_with_score"):
        docs = docsearch.similarity_search(query, k=k)
        if not docs:
            return [], True
        return docs, False

    # Sinon, on a similarity_search_with_score
    docs_with_scores = docsearch.similarity_search_with_score(query, k=k)
    if not docs_with_scores:
        # => Pas de doc => fallback
        return [], True

    # On parcourt pour voir si au moins un doc est au-dessus du seuil
    relevant_docs = []
    all_below = True
    for doc, score in docs_with_scores:
        if score < threshold:
            all_below = False
        relevant_docs.append(doc)

    if all_below:
        return [], True
    else:
        return relevant_docs, False


def _ollama_direct(ollama_llm: OllamaLLM, prompt: str) -> str:
    """Appel direct du LLM (hors RAG)"""
    try:
        return ollama_llm(prompt)
    except Exception as e:
        return f"Erreur lors de l'appel direct à Ollama : {e}"
