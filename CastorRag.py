
#installing the dependecies 
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings

#loading the variables from the .env file

load_dotenv()

# Initialize the free Gemini model tier
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
#loading the document

loader = PyPDFLoader("castoil pdf.pdf")
document = loader.load()

#chunking the documents
test_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50,
    separators=["\n\n", "\n", ". ", " "]
)
chunks = test_splitter.split_documents(document)
#embeddings the documents


#load the embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#load chromaDB

PERSIST_DIRECTORY ="./chroma_db"

db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings,
)
# Add your document chunks
db.add_documents(chunks)


# Define the prompt template
prompt_template = PromptTemplate.from_template("""
You are a question-answering assistant.

Answer the question using ONLY the information provided in the context below.

If the answer cannot be found in the context, say:
"I don't know based on the provided document."

Do not use outside knowledge.
Do not make up information.

Keep the answer concise.

Context:
{context}

Question:
{question}

Answer:
""")


def ask_question(query):

    retriever = db.as_retriever(
        search_kwargs={"k": 2}
    )

    docs = retriever.invoke(query)

    if docs:
        context = "\n\n".join(
            doc.page_content for doc in docs
        )
    else:
        context = "No documents were retrieved."

    prompt = prompt_template.format(
        context=context,
        question=query
    )

    response = llm.invoke(prompt)

    # Clean response
    content = response.content

    if isinstance(content, list):
        clean_answer = "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict)
        )
    else:
        clean_answer = content

    return clean_answer

print("\n📄 PDF Chatbot is ready!")
print("Ask a question about your PDF.")
print("Type 'exit' or 'quit' to stop.")

while True:
    user_question = input("\nYour question: ").strip()

    if user_question.lower() in ["exit", "quit"]:
        print("Goodbye! 👋")
        break

    answer = ask_question(user_question)

    print(f"\nAnswer: {answer}")