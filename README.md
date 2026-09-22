# 🧪 Castor Oil AI Agent

An AI-powered **Retrieval-Augmented Generation (RAG) agent** for answering questions related to castor oil production, extraction, processing, and chemical engineering.

The project combines **LangGraph**, **Google Gemini**, **ChromaDB**, **Hugging Face embeddings**, and **Model Context Protocol (MCP)** to build an agent that can retrieve information from a castor oil technical document and use it to answer domain-specific questions.

> **Note:** This project is an educational prototype and is not intended to replace professional chemical engineering advice.

---

## 🚀 Project Overview

Large Language Models can generate useful answers, but they may also produce information that is not contained in a specific technical document.

This project addresses that problem by using **Retrieval-Augmented Generation (RAG)**.

Instead of relying only on the language model's internal knowledge, the system:

1. Receives a user's question.
2. Determines whether the question is related to the supported domain.
3. Retrieves relevant information from the castor oil knowledge base.
4. Passes the retrieved context to the language model.
5. Generates an answer based on the retrieved information.
6. Uses MCP tools for additional functionality such as unit conversion.

### Architecture

```text
                    User Question
                          │
                          ▼
                  ┌───────────────┐
                  │   LangGraph   │
                  │  Agent Router │
                  └───────┬───────┘
                          │
                ┌─────────┴─────────┐
                │                   │
          Domain Related        Off Topic
                │                   │
                ▼                   ▼
        ┌───────────────┐    ┌─────────────┐
        │   Retriever   │    │   Response  │
        └───────┬───────┘    └─────────────┘
                │
                ▼
          ┌─────────────┐
          │  ChromaDB   │
          │ Vector Store│
          └──────┬──────┘
                 │
                 ▼
        Relevant Document
             Chunks
                 │
                 ▼
        ┌─────────────────┐
        │   Gemini LLM    │
        │  Answer Agent   │
        └────────┬────────┘
                 │
                 ▼
             Final Answer

                 +
                 
          ┌─────────────┐
          │ MCP Server  │
          │             │
          │ Unit Tools  │
          └─────────────┘
```

---

## ✨ Features

* 📚 **Document-based question answering**
* 🔎 **Semantic retrieval using vector embeddings**
* 🧠 **Retrieval-Augmented Generation (RAG)**
* 🤖 **LangGraph agent workflow**
* 🚦 **Question routing / domain classification**
* 🧪 **Castor oil and chemical engineering knowledge**
* 🔢 **Unit conversion through MCP tools**
* 💾 **Persistent ChromaDB vector store**
* 🛡️ **Off-topic question handling**
* 🔌 **Model Context Protocol (MCP) integration**
* 💬 Interactive command-line interface

---

## 🛠️ Technologies Used

| Technology            | Purpose                                |
| --------------------- | -------------------------------------- |
| Python                | Core programming language              |
| LangChain             | RAG components and document processing |
| LangGraph             | Agent workflow and routing             |
| Google Gemini         | Large Language Model                   |
| ChromaDB              | Vector database                        |
| Hugging Face          | Text embedding model                   |
| Sentence Transformers | Document embeddings                    |
| MCP                   | Tool integration                       |
| PyPDF                 | PDF document loading                   |
| dotenv                | Environment variable management        |

---

## 📂 Project Structure

```text
Castor-Oil-AI-Agent/
│
├── CastorRag.py
├── RagAgent.py
├── main.py
│
├── castoil pdf.pdf
│
├── chroma_db/
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

### File Descriptions

#### `CastorRag.py`

Contains the core RAG functionality.

Responsibilities include:

* Loading the castor oil PDF
* Splitting the document into chunks
* Creating embeddings
* Loading ChromaDB
* Retrieving relevant document chunks
* Generating answers using Gemini

---

#### `RagAgent.py`

Contains the LangGraph-based agent workflow.

The agent is responsible for:

* Receiving user questions
* Classifying questions
* Routing questions
* Retrieving relevant context
* Generating responses
* Handling unsupported questions

---

#### `main.py`

Acts as the main entry point for running the application.

---

#### `castoil pdf.pdf`

The technical document used as the knowledge source for the RAG system.

---

#### `chroma_db/`

Persistent vector database containing the document embeddings used during retrieval.

---

## 🔧 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` yet, install the major dependencies:

```bash
pip install langchain langchain-community langchain-chroma langchain-google-genai langgraph chromadb sentence-transformers pypdf python-dotenv mcp
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Do **not** commit your `.env` file to GitHub.

Your `.gitignore` should contain:

```gitignore
.env
.venv/
__pycache__/
chroma_db/
```

---

## ▶️ Running the Project

After activating the virtual environment, run:

```bash
python main.py
```

Depending on the implementation, you can then enter questions through the terminal.

Example:

```text
Your question: What are the methods used for drying castor seeds?
```

The system retrieves relevant information from the knowledge base and generates a response.

---

## 🔍 How the RAG Pipeline Works

### Step 1 — Document Loading

The castor oil PDF is loaded using a PDF document loader.

```text
PDF
 ↓
Document Loader
```

### Step 2 — Text Splitting

The document is divided into smaller chunks.

The current implementation uses approximately:

```text
Chunk size: 500
Chunk overlap: 50
```

This allows the retrieval system to find relevant sections without processing the entire document for every question.

### Step 3 — Embeddings

Each document chunk is converted into a numerical vector using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

These vectors represent the semantic meaning of the text.

### Step 4 — Vector Storage

The embeddings are stored in **ChromaDB**.

```text
Document Chunks
      ↓
Embeddings
      ↓
ChromaDB
```

### Step 5 — Retrieval

When a user asks a question, the system searches the vector database for the most semantically relevant chunks.

For example:

```text
Question:
"What temperature is used to dry castor seeds?"

             ↓

       Semantic Search

             ↓

Relevant document chunks
```

### Step 6 — Generation

The retrieved context is provided to Gemini along with the user's question.

The model is instructed to answer using the retrieved information.

---

## 🧠 LangGraph Agent

The project uses LangGraph to structure the AI workflow as a state-based graph.

A simplified workflow is:

```text
START
  │
  ▼
Question Classification
  │
  ├───────────────┐
  │               │
Relevant        Not Relevant
  │               │
  ▼               ▼
Retriever       Off-topic
  │              Response
  ▼
Context
  │
  ▼
LLM
  │
  ▼
END
```

This approach makes the application easier to extend with additional tools and reasoning steps.

---

## 🔌 MCP Integration

The project also explores **Model Context Protocol (MCP)** for connecting the AI agent to external tools.

One example is a unit conversion tool.

Supported conversions include:

```text
kg ↔ g
m ↔ cm
mg ↔ g
L ↔ mL
```

Example:

```text
Input:
Convert 5 kg to grams

Output:
5000 g
```

MCP allows additional tools to be added without putting all of the functionality directly inside the language model.

---

## 💡 Example Questions

The system can be used for questions such as:

```text
What methods are used to extract castor oil?

What is the purpose of drying castor seeds?

What temperature is used during drying?

What equipment is used for oil extraction?

Explain the Soxhlet extraction process.

What solvents are used during extraction?

What is esterification?

What is transesterification?

What is the difference between esterification and transesterification?
```

The exact answers depend on the information available in the indexed document.

---

## 🛡️ Reducing Hallucinations

One of the main goals of this project is to reduce unsupported answers.

The RAG system provides the language model with retrieved document context and instructs it to use that context when answering.

If the required information cannot be found in the retrieved context, the agent can respond that the information is not available in the provided document rather than presenting unsupported information as fact.

However, RAG does **not** completely eliminate hallucinations. Retrieval quality, chunking, embeddings, prompting, and model behavior can all affect the final response.

---

## 📈 Future Improvements

Possible improvements include:

* [ ] Improve retrieval accuracy
* [ ] Add metadata filtering
* [ ] Implement hybrid search
* [ ] Add reranking
* [ ] Improve citation/source tracking
* [ ] Add conversation memory
* [ ] Add more chemical engineering tools
* [ ] Expand MCP tool support
* [ ] Add a web interface
* [ ] Add evaluation metrics for RAG
* [ ] Add automated tests
* [ ] Add Docker support
* [ ] Improve error handling
* [ ] Add document upload functionality
* [ ] Support multiple technical documents

---

## ⚠️ Limitations

This project is currently an **educational prototype**.

The system's responses depend on:

* The quality of the source document
* Document chunking
* Embedding quality
* Retrieval accuracy
* LLM behavior
* Prompt design

The system should therefore not be treated as a replacement for professional chemical engineering expertise.

---

## 🎯 Project Goal

The goal of this project is to explore how **RAG, agentic workflows, vector databases, LLMs, and MCP tools** can be combined to build a domain-specific AI assistant.

The project also serves as a practical implementation of concepts in:

* Artificial Intelligence
* Machine Learning
* Natural Language Processing
* Retrieval-Augmented Generation
* AI Agents
* Vector Databases
* Chemical Engineering
* Tool Calling
* Model Context Protocol

---

## 👩🏽‍💻 Author

**Chiamaka Egbulefu**

AI/ML Developer

Interested in building AI systems that combine machine learning models, LLMs, RAG pipelines, and agentic workflows.

---

## 📄 Disclaimer

This project is provided for **educational and experimental purposes only**.

Information generated by the AI system should be independently verified before being used for research, engineering decisions, laboratory procedures, or other real-world applications.
