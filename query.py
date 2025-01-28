from langchain import hub
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.prompts import ChatPromptTemplate

from scrapping import vector_store
from setup import llm


# Define prompt for question-answering
# prompt = hub.pull("rlm/rag-prompt")

prompt = ChatPromptTemplate.from_messages([
    ("system", "tu es un assistant IA prêt à répondre à toutes mes questions concernant le manuel de gestion de l'UQAC. "
               "Utilises les éléments de contexte suivants pour répondre à la question. Si tu ne connais pas la réponse, dis-le simplement."
               "Essaie de répondre de manière concise et précise."),
    ("human", "{question}"),
    ("system", "Voici les éléments de contexte our répondre à la question : {context}"
               "Ainsi aue les messages précédents : {history}"),
])

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    history: List[dict]


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    try:
        history_content = "\n".join(f"Question: {h['question']}\nRéponse: {h['answer']}" for h in state["history"])
    except KeyError:
        history_content = "Aucune question n'a été posée jusqu'à présent."
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])

    messages = prompt.invoke({"question": state["question"], "context": docs_content, "history": history_content})
    response = llm.invoke(messages)

    return {"answer": response}


def update_history(state: State):
    try:
        updated_history = state["history"] + [{"question": state["question"], "answer": state["answer"]}]
    except KeyError:
        updated_history = [{"question": state["question"], "answer": state["answer"]}]
    return {"history": updated_history}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate, update_history])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

print("graph compiled")