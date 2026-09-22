import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings



# Load environment variables

load_dotenv()


# Configuration

PDF_PATH = "castoil pdf.pdf"
PERSIST_DIRECTORY = "./chroma_db"


# Load embeddings

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# Load Chroma

print("Loading Chroma database...")

db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings,
)

print("Chroma database loaded.")


# Check if database is empty

print("Checking Chroma database...")

existing_documents = db.get(limit=1)


if not existing_documents["ids"]:

    print("Chroma is empty.")
    print("Loading PDF...")

    loader = PyPDFLoader(PDF_PATH)

    document = loader.load()

    print(f"PDF loaded. Pages: {len(document)}")


        # Split PDF into chunks

    print("Splitting document...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "]
    )

    chunks = text_splitter.split_documents(document)

    print(f"Created {len(chunks)} chunks.")


    # Add documents to Chroma

    print("Adding documents to Chroma...")

    db.add_documents(chunks)

    print("Documents added successfully.")

else:

    print("Existing documents found.")
    print("Skipping PDF processing.")
    print("Skipping embedding.")
    print("Skipping db.add_documents().")


# Create retriever

retriever = db.as_retriever(
    search_kwargs={"k": 2}
)


# LLM

llm = ChatGroq(
    model="openai/gpt-oss-120b")


# Prompt

prompt_template = PromptTemplate.from_template(
    """
You are a question-answering assistant.

Answer the question using ONLY the information
provided in the context below.

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
"""
)


# Retrieve context

def retrieve_context(query: str) -> str:

    docs = retriever.invoke(query)

    if not docs:
        return "No relevant information was found."

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# Ask question

def ask_question(query: str):

    context = retrieve_context(query)

    prompt = prompt_template.format(
        context=context,
        question=query
    )

    response = llm.invoke(prompt)

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


# Standalone chatbot

if __name__ == "__main__":

    print("\n📄 PDF Chatbot is ready!")
    print("Ask a question about your PDF.")
    print("Type 'exit' or 'quit' to stop.")

    while True:

        user_question = input("\nYour question: ").strip()

        if user_question.lower() in ["exit", "quit"]:

            print("Goodbye! 👋")
            break

        answer = ask_question(user_question)

        print("\nAnswer:")
        print(answer)