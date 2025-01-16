from query import graph

question = str(input("Enter your question: "))
if question is None:
    question = "Quels sont les objectifs de la politique de développement durable de l'UQAC ?"

print("answering ...")
response = graph.invoke({"question": question})
print(response["answer"])
