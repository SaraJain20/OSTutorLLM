# OSTutorLLM — AI-Powered Operating Systems Tutor

OSTutorLLM is an AI-powered learning assistant designed to help students understand Operating Systems concepts through an interactive question-answering experience. It uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from Operating Systems study materials and generate context-aware responses.

## Features

- **Interactive learning:** Ask questions about Operating Systems concepts in natural language.
- **PDF-based knowledge retrieval:** Retrieve relevant information from a collection of Operating Systems textbooks and study materials.
- **Retrieval-Augmented Generation (RAG):** Ground responses in retrieved document content.
- **Semantic search:** Use sentence-transformer embeddings to find relevant text passages.
- **FAISS vector search:** Efficiently search document embeddings for relevant context.
- **AI-generated responses:** Use a language model through Groq to answer questions.
- **MongoDB integration:** Store chat history and quiz attempts in MongoDB Atlas.

## Technology Stack

| Component | Technology |
|---|---|
| User interface | Streamlit |
| Programming language | Python |
| LLM integration | LangChain and Groq |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector database/index | FAISS |
| Document processing | PyPDF |
| Database | MongoDB Atlas |
| Environment configuration | Python Dotenv |

## How It Works

1. **Document loading:** Operating Systems learning materials are loaded from PDF documents.
2. **Text processing:** Documents are split into smaller chunks for retrieval.
3. **Embedding generation:** Text chunks are converted into vector embeddings.
4. **Vector indexing:** FAISS indexes the embeddings for semantic similarity search.
5. **Question processing:** A student's question is used to retrieve relevant document chunks.
6. **Answer generation:** The retrieved context is passed to the language model to generate a response.
7. **Data storage:** Chat history and quiz attempts can be stored in MongoDB Atlas.

## Project Structure

```text
OSTutorLLM/
├── app.py
├── test_mongodb.py
├── requirements.txt
├── .gitignore
├── .env                 # Local secrets; not committed to Git
├── .venv/               # Local virtual environment; not committed
└── DATA/                # Local PDF learning materials; excluded from Git
```

*Note: `.env`, `.venv/`, and the PDF files in `DATA/` are intended to remain local and are not included in the public repository.*

## Getting Started

### Prerequisites

- Python installed on your computer
- Git
- A Groq API key
- A MongoDB Atlas connection URI if using the database features
- The required Operating Systems PDF materials

### 1. Clone the Repository

```bash
git clone https://github.com/SaraJain20/OSTutorLLM.git
cd OSTutorLLM
```

### 2. Create a Virtual Environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use the environment's Python executable directly instead.

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a file named `.env` in the project's root directory. Add the environment variable names expected by your `app.py`, using your own credentials:

```env
GROQ_API_KEY=your_groq_api_key
MONGODB_URI=your_mongodb_connection_uri
```

Replace the example values with valid credentials. If your application uses different variable names, use the exact names defined in your code.

**Security:** Never commit or publish your `.env` file, API keys, or database credentials.

### 5. Add the Learning Materials

Place the required Operating Systems PDF files in the local `DATA/` directory. These files are excluded from the Git repository, so each user must obtain and add the materials separately. Only use PDFs you have permission to use.

### 6. Run the Application

```bash
python -m streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser to use the application.

## Database

The application integrates with MongoDB Atlas for storing application data, including chat history and quiz attempts. Database functionality requires a valid connection URI and appropriate database permissions.

## Security and Privacy

- Keep API keys and database credentials in local environment variables.
- Do not commit `.env` or virtual environment folders.
- Do not publish private credentials or sensitive data.
- Configure MongoDB Atlas network access and database permissions carefully.

## Current Scope

OSTutorLLM is a learning assistant focused on Operating Systems. Its responses depend on the available learning materials, retrieval quality, model behavior, and the features implemented in the application.

## Future Improvements

- Add topic-wise quizzes and practice questions.
- Improve answer citations and document references.
- Expand the learning-material collection.
- Improve error handling and user feedback.
- Add progress tracking and a more detailed learning dashboard.

## Author

**Sara Jain**
**Priya Pushp**

GitHub: [@SaraJain20](https://github.com/SaraJain20)
GitHub: [@priya28-pushp](https://github.com/priya28-pushp)
## License

No license has been specified yet. If you want others to reuse, modify, or distribute this project, consider adding an appropriate open-source license.
