import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import cassio
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv

load_dotenv()
astra_vector_store=None

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_chunks(text):
   # We need to split the text using Character Text Split such that it sshould not increse token size
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 600,
        chunk_overlap  = 150,
        length_function = len,
    )
    chunks = text_splitter.split_text(text)
    # chunks[:50]
    return chunks

def get_vector_store(text_chunks):
    application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
    astra_db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
    ASTRA_DB_REGION=os.getenv("ASTRA_DB_REGION")
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    embedding = OpenAIEmbeddings(model="text-embedding-3-large",openai_api_key=OPENAI_API_KEY)

    cassio.init(token=application_token, database_id=ASTRA_DB_ID)

    astra_vector_store = Cassandra(
        embedding=embedding,
        table_name="pdfnotes",
        session=None,
        keyspace="default_keyspace"
    )

    astra_vector_store.add_texts(text_chunks)

    print("Inserted %i headlines." % len(text_chunks))

    astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)
    # llm = ChatOpenAI(
    #     model="gpt-3.5-turbo-16k",  # <-- Important to avoid 4097 token limit
    #     openai_api_key=OPENAI_API_KEY
    # )

    # query_text = "Explain me about the key consideration while determining Competence & Capability"

    # answer = astra_vector_index.query(query_text, llm=llm).strip()
    # print("ANSWER: \"%s\"\n" % answer)

def user_input(user_question):
    # os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # 2. Initialize the ChatOpenAI model
    llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0.7)

    # 3. Set up memory to store chat history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    embedding = OpenAIEmbeddings(model="text-embedding-3-large",openai_api_key=OPENAI_API_KEY)

    application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
    cassio.init(token=application_token, database_id=ASTRA_DB_ID, keyspace="default_keyspace")

    astra_vector_store = Cassandra(
        embedding=embedding,
        table_name="pdfnotes",
        session=None,
        keyspace="default_keyspace"
    )

    # Setup retriever
    retriever = astra_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})

    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )

    result = rag_chain({"question": user_question})

    print("Answer:", result["answer"])
    # # Optional: print source docs
    # for doc in result["source_documents"]:
    #     print("---")
    #     print(doc.page_content)

    st.write("Reply: ", result["answer"])

def main():
    
    
    st.set_page_config("Chat CA-BOT")
    st.header("Chat with CA-Notes using our Agent💁")

    user_question = st.text_input("Ask a Question from the your Notes")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()