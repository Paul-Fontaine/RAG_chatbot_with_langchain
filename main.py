from query import graph

# question = str(input("Enter your question: "))
# if question == "0":
question = "Quels sont les objectifs de la politique de développement durable de l'UQAC ?"
print("question:", question)

print("answering ...")
response = graph.invoke({"question": question})
print(response["answer"])
print("source:", response['context'][0].metadata['source'])
