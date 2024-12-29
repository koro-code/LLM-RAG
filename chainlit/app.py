import os
from typing import List

import chainlit as cl
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.memory import ChatMessageHistory, ConversationBufferMemory

# Définir la clé API OpenAI
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

@cl.on_chat_start
async def on_chat_start():
    files = None

    # Attendre que l'utilisateur télécharge un fichier
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

    # Diviser le texte en segments
    texts = text_splitter.split_text(text)

    # Créer des métadonnées pour chaque segment
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Créer un store vectoriel Chroma avec un persist_directory
    embeddings = OpenAIEmbeddings()
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts,
        embeddings,
        metadatas=metadatas,
        persist_directory="./chroma_db"  # Spécifiez un répertoire persistant
    )
    docsearch.persist()  # Sauvegarder la base de données

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Créer une chaîne qui utilise le store vectoriel Chroma
    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    # Informer l'utilisateur que le système est prêt
    msg.content = f"Traitement de `{file.name}` terminé. Vous pouvez maintenant poser vos questions !"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()

    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Créer l'élément de texte référencé dans le message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name, display="side")
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources : {', '.join(source_names)}"
        else:
            answer += "\nAucune source trouvée"

    await cl.Message(content=answer, elements=text_elements).send()