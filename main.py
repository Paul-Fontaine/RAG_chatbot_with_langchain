from query import graph


def ask_question(question: str):
    print("answering ...")
    response = graph.invoke({"question": question})
    answer = response["answer"]
    sources = set([doc.metadata['source'] for doc in response['context'] if doc.metadata['source'] is not None])
    reponse = (f"réponse :\n"
               f"{answer}\n"
               f"sources : \n"
               f"{"\n".join(sources)}\n")
    print(reponse)

while True:
    # question = str(input("question :\n"))
    question = "Quelle est la langue de travail de l'UQAC ?"
    if question == "exit":
        break
    ask_question(question)

# Quels sont les objectifs de la politique de développement durable de l'UQAC ?
# Qui est responsable de son application ?
# Quelle est la définition du développement durable  dans le manuel de gestion de l'uqac?
# Quelles sont les principales fonctions du Comité de mise en œuvre du développement durable ?
