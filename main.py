from query import graph

# question = str(input("Enter your question: "))
# if question == "0":
question = "Quels sont les objectifs de la politique de développement durable de l'UQAC ?"
print("question:", question)

print("answering ...")
response = graph.invoke({"question": question})
answer = response["answer"]
sources = set([doc.metadata['source'] for doc in response['context'] if doc.metadata['source'] is not None])
print("réponse :\n", answer)
print("\nsources : \n", "\n".join(sources))
