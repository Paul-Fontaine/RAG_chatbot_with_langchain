from query import graph
import streamlit as st

def ask_question(question: str):
    print("answering ...")
    response = graph.invoke({"question": question})
    answer = response["answer"]
    sources = set([doc.metadata['source'] for doc in response['context'] if doc.metadata['source'] is not None])

    sources_text = "\n".join(sources)  

    reponse = f"\n{answer}\n\n{sources_text}\n"  

    print(reponse)  

    return reponse

st.title("💬 RAG Manuel de Gestion UQAC")

# Initialiser l'historique des messages si non défini
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Gestion de la nouvelle question
if user_input := st.chat_input("Posez une question..."):

    # Ajouter et afficher immédiatement la question
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Obtenir la réponse du LLM
    bot_reply = ask_question(user_input)

    # Ajouter et afficher immédiatement la réponse
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.write(bot_reply)
        
