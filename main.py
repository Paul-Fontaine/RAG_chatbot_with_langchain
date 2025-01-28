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
print("\nsources : \n", "\n".join(sources), "\n")

question2 = "À qui s'applique cette politique et qui est le responsable de son application ?"
print("question:", question2)
response2 = graph.invoke({"question": question2})
answer2 = response2["answer"]
sources2 = set([doc.metadata['source'] for doc in response2['context'] if doc.metadata['source'] is not None])
print("réponse :\n", answer2)
print("\nsources : \n", "\n".join(sources2), "\n")

question3 = "Reformule ta réponse précédente."
print("question:", question3)
response3 = graph.invoke({"question": question3})
answer3 = response3["answer"]
sources3 = set([doc.metadata['source'] for doc in response3['context'] if doc.metadata['source'] is not None])
print("réponse :\n", answer3)
print("\nsources : \n", "\n".join(sources3), "\n")
