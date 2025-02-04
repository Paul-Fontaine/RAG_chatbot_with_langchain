from query import graph

def ask_question(question: str):
    print("answering ...")
    response = graph.invoke({"question": question})
    answer = response["answer"]
    sources = set([doc.metadata['source'] for doc in response['context'] if doc.metadata['source'] is not None])

    sources_text = "\n".join(sources)  # Construire d'abord la chaîne de sources séparément

    reponse = f"réponse :\n{answer}\n\nsources :\n{sources_text}\n"  # Insérer la variable dans la f-string

    print(reponse)

while True:
    question = str(input("question :\n"))
    if question == "exit":
        break
    ask_question(question)
