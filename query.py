from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from setup import llm, vector_store

# Pré-prompt pour le chatbot
preprompt = ("Tu es un assistant IA RAG prêt à répondre à toutes les questions concernant le manuel de gestion de l'UQAC. "
             "Des éléments de contexte te seront fournis à chaque question, appuies toi dessus pour répondre.")
chat_memory = InMemoryChatMessageHistory()
chat_memory.add_message(SystemMessage(preprompt))

# Définir l'état de l'application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Étapes de l'application
def retrieve(state: State):
    """
    Récupère les documents pertinents depuis Chroma en fonction de la question.
    """
    retrieved_docs = vector_store.similarity_search(state["question"], k=4)
    return {"context": retrieved_docs}

def generate(state: State):
    """
    Génère une réponse basée sur les documents récupérés et la question de l'utilisateur.
    """
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])

    question = HumanMessage(state["question"])
    context = SystemMessage("Voici les éléments de contexte : \n" + docs_content)
    messages = [
        *chat_memory.messages,
        question,
        context
    ]
    # Appelle le modèle Mistral pour générer la réponse
    response = llm.invoke(messages)

    chat_memory.add_message(question)
    chat_memory.add_message(context)
    chat_memory.add_message(AIMessage(response))

    return {"answer": response}

# Compilation de l'application
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()
