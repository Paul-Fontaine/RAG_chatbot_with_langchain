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
    ("system", "Voici les éléments de contexte our répondre à la question : {context}"),
])

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

print("graph compiled")