from typing import Any, Dict, List
import psycopg2
from psycopg2.extras import RealDictCursor

from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory import ChatMessageHistory
from pydantic import Field

class PostgresConversationBufferMemory(BaseChatMemory):
    """
    Memory stockant et récupérant les messages (utilisateur/assistant)
    dans PostgreSQL.
    """

    session_id: str
    connection: Any = Field(...)
    chat_memory: ChatMessageHistory = Field(default_factory=ChatMessageHistory)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        print(f"[DEBUG:Memory] __init__ called for session_id={self.session_id}")
        self.chat_memory = self._load_history()
        print(f"[DEBUG:Memory] Loaded history => {self.chat_memory.messages}")

    def _load_history(self) -> ChatMessageHistory:
        """Lit la table conversation_history en base et construit un ChatMessageHistory."""
        print(f"[DEBUG:Memory] _load_history() for session_id={self.session_id}")
        history = ChatMessageHistory()
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cur:
                query = """
                    SELECT role, content
                    FROM conversation_history
                    WHERE session_id = %s
                    ORDER BY created_at ASC
                """
                cur.execute(query, (self.session_id,))
                rows = cur.fetchall()
                print(f"[DEBUG:Memory] Found {len(rows)} rows in conversation_history for session_id={self.session_id}.")

                for row in rows:
                    role = row["role"]
                    content = row["content"]
                    print(f"[DEBUG:Memory] row: role={role}, content={content[:50]}...")
                    if role == "user":
                        history.add_user_message(content)
                    elif role == "assistant":
                        history.add_ai_message(content)

        except Exception as e:
            print(f"[ERROR:Memory] _load_history exception: {e}")
        return history

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        print(f"[DEBUG:Memory] load_memory_variables called. chat_history length={len(self.chat_memory.messages)}")
        return {"chat_history": self.chat_memory.messages}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        # Pour usage sync (ConversationBufferMemory standard), si nécessaire
        pass

    async def asave_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """
        Méthode asynchrone appelée par les chaînes (ConversationalRetrievalChain, LLMChain, etc.)
        pour enregistrer la conversation dans la DB.
        """
        print(f"[DEBUG:Memory] asave_context called. inputs={inputs}, outputs={outputs}, session_id={self.session_id}")

        # Filtrer "source_documents" s'il existe
        filtered_outputs = {k: v for k, v in outputs.items() if k != "source_documents"}

        # CHANGEMENT 1 : Récupérer la question et la réponse
        user_message = inputs.get("question") or inputs.get("input", "")
        # On va chercher soit 'answer', soit 'text'
        ai_message = filtered_outputs.get("answer") or filtered_outputs.get("text", "")

        # MàJ mémoire en RAM
        self.chat_memory.add_user_message(user_message)
        self.chat_memory.add_ai_message(ai_message)

        # Écriture en BDD
        try:
            with self.connection.cursor() as cur:
                insert_query = """
                    INSERT INTO conversation_history (session_id, role, content)
                    VALUES (%s, %s, %s)
                """
                # user
                cur.execute(insert_query, (self.session_id, "user", user_message))
                # assistant
                cur.execute(insert_query, (self.session_id, "assistant", ai_message))
                self.connection.commit()
                print("[DEBUG:Memory] asave_context committed to DB.")
        except Exception as e:
            print(f"[ERROR:Memory] asave_context exception: {e}")

    def clear(self):
        print(f"[DEBUG:Memory] clear() called for session_id={self.session_id}")
        try:
            with self.connection.cursor() as cur:
                cur.execute("DELETE FROM conversation_history WHERE session_id = %s", (self.session_id,))
                self.connection.commit()
            self.chat_memory = ChatMessageHistory()
            print("[DEBUG:Memory] History cleared.")
        except Exception as e:
            print(f"[ERROR:Memory] clear exception: {e}")

    @property
    def memory_variables(self) -> List[str]:
        return ["chat_history"]
