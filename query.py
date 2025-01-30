from langchain import hub
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


from setup import llm, vector_store


# Define prompt for question-answering
# prompt = hub.pull("rlm/rag-prompt")

# todo: améliorer ce prompt
# todo: utiliser https://python.langchain.com/docs/how_to/chatbots_memory/ pour la mémoire des questions précédentes
preprompt = ("Tu es un assistant IA RAG prêt à répondre à toutes les questions concernant le manuel de gestion de l'UQAC."
             "Des éléments de contexte te seront fournis à chaque question, appuies toi dessus pour répondre.")

chat_memory = InMemoryChatMessageHistory()
chat_memory.add_message(SystemMessage(preprompt))

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"], k=4)
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])

    question = HumanMessage(state["question"])
    context = SystemMessage("Voici les éléments de contexte : \n" + docs_content)
    messages = [
        *chat_memory.messages,
        question,
        context
    ]
    response = llm.invoke(messages)

    chat_memory.add_message(question)
    chat_memory.add_message(context)
    chat_memory.add_message(AIMessage(response))

    return {"answer": response}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()
