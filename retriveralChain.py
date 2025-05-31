from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain

llm = ChatOpenAI()

template = """Answer the following question based on the provided context
<context>
{context}
</context>

Question:{input}
"""

#This creates a chain that will combine documents and use the provided template and language model to generate responses.
prompt = ChatPromptTemplate.from_template(template)
document_chain = create_stuff_documents_chain(llm,prompt)

# Example usage
context = "The sun rises in the east and sets in the west."
question = "Which direction does the sun set?"

# Convert context into a Document object
documents = [Document(page_content=context)]

# Get the answer from the document chain
answer = document_chain.invoke({"input": question, "context": documents})
print(answer)