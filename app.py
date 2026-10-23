import os
import json
import re
from datetime import datetime, timezone

import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# =========================================================
# 1. CONFIGURATION
# =========================================================

load_dotenv()

st.set_page_config(
    page_title="OSTutorLLM",
    page_icon="🐧",
    layout="wide"
)

st.title("🐧 OSTutorLLM: Interactive OS Tutor")
st.caption(
    "Learn Operating Systems, practise quizzes, and track your progress."
)

# Required values must be present in your .env file.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is missing from your .env file.")
    st.stop()

if not MONGODB_URI:
    st.error("MONGODB_URI is missing from your .env file.")
    st.stop()


# =========================================================
# 2. CONNECT TO MONGODB
# =========================================================

@st.cache_resource
def connect_mongodb(uri):
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)

    # Check that MongoDB is reachable.
    client.admin.command("ping")

    database = client["ostutor_db"]

    chat_collection = database["chat_history"]
    quiz_collection = database["quiz_attempts"]

    return client, chat_collection, quiz_collection


try:
    mongo_client, chat_collection, quiz_collection = (
        connect_mongodb(MONGODB_URI)
    )
    mongo_connected = True

except Exception as error:
    mongo_connected = False
    st.error(f"Could not connect to MongoDB: {error}")
    st.stop()


# =========================================================
# 3. STUDENT USERNAME
# =========================================================

st.sidebar.header("Student Profile")

username = st.sidebar.text_input(
    "Student username",
    value=st.session_state.get("username", "test_student")
).strip()

if not username:
    st.warning("Enter a student username to continue.")
    st.stop()

st.session_state["username"] = username

st.sidebar.info(
    "Your chat and quiz records are saved under this username."
)

st.sidebar.success("MongoDB connected")
st.sidebar.caption("Database: ostutor_db")


# =========================================================
# 4. LOAD SYLLABUS PDFs AND BUILD RAG DATABASE
# =========================================================

@st.cache_resource
def setup_rag_database():
    data_folder = "data"

    os.makedirs(data_folder, exist_ok=True)

    loader = PyPDFDirectoryLoader(data_folder)
    documents = loader.load()

    if not documents:
        return None, None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        return None, None

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = FAISS.from_documents(chunks, embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})

    return vector_db, retriever


with st.spinner("Loading Operating Systems syllabus..."):
    try:
        vector_db, retriever = setup_rag_database()
    except Exception as error:
        st.error(f"Could not load the syllabus PDFs: {error}")
        st.stop()

if retriever is None:
    st.warning(
        "No readable PDFs were found in the data folder. "
        "Add your OS syllabus PDFs and restart the app."
    )


# =========================================================
# 5. INITIALIZE GROQ AI MODEL
# =========================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.3,
    api_key=GROQ_API_KEY
)


# =========================================================
# 6. CHAT TUTOR WITH RAG
# =========================================================

def format_docs(documents):
    return "\n\n".join(
        document.page_content for document in documents
    )


system_prompt = """
You are OSTutorLLM, an interactive Operating Systems professor.

Use the syllabus context below to answer the student's question.

Syllabus context:
{context}

Rules:
1. Explain concepts in simple language.
2. Keep the initial explanation concise.
3. Use examples when useful.
4. Do not invent information that is not supported by the context.
5. If the context does not contain the answer, say so honestly.
6. End with one relevant follow-up question, mini-quiz,
   or practice task.
"""

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

chat_chain = chat_prompt | llm | StrOutputParser()


def save_chat_message(user, role, content):
    """Save one chat message in MongoDB."""
    chat_collection.insert_one({
        "student": user,
        "role": role,
        "content": content,
        "created_at": datetime.now(timezone.utc)
    })


def get_chat_history(user):
    """Retrieve a student's previous chat messages."""
    records = chat_collection.find(
        {"student": user}
    ).sort("created_at", 1)

    return [
        {
            "role": record["role"],
            "content": record["content"]
        }
        for record in records
    ]


def generate_chat_answer(question):
    """Retrieve syllabus context and generate an answer."""
    if retriever is None:
        return (
            "I cannot access the syllabus because no readable PDFs "
            "were found in the data folder. Please add your OS PDFs."
        ), []

    retrieved_documents = retriever.invoke(question)
    context = format_docs(retrieved_documents)

    answer = chat_chain.invoke({
        "context": context,
        "input": question
    })

    return answer, retrieved_documents


# =========================================================
# 7. QUIZ GENERATION
# =========================================================

def generate_quiz(topic, number_of_questions):
    """
    Generate MCQs as JSON.
    The doubled braces below are intentional: they prevent
    Python f-strings from interpreting JSON braces as variables.
    """

    quiz_prompt = f"""
Generate exactly {number_of_questions} multiple-choice questions
about this Operating Systems topic: {topic}

Return ONLY valid JSON using this structure:

{{
    "questions": [
        {{
            "question": "Question text",
            "options": {{
                "A": "First option",
                "B": "Second option",
                "C": "Third option",
                "D": "Fourth option"
            }},
            "correct_answer": "A",
            "explanation": "Explain the correct answer."
        }}
    ]
}}

Requirements:
- Generate exactly {number_of_questions} questions.
- Every question must have exactly four options: A, B, C, D.
- correct_answer must be one of A, B, C, or D.
- Make the questions factually correct and educational.
- Use clear language appropriate for a college student.
- Return JSON only. Do not use Markdown code fences.
"""

    response = llm.invoke(quiz_prompt).content

    if isinstance(response, list):
        response = "".join(
            part.get("text", "") if isinstance(part, dict)
            else str(part)
            for part in response
        )

    # Remove optional Markdown code fences.
    response = re.sub(
        r"^\s*```(?:json)?\s*|\s*```\s*$",
        "",
        str(response),
        flags=re.IGNORECASE
    ).strip()

    # Parse JSON.
    try:
        quiz_data = json.loads(response)
    except json.JSONDecodeError:
        # Attempt to extract a JSON object if extra text was returned.
        match = re.search(r"\{.*\}", response, flags=re.DOTALL)

        if not match:
            raise ValueError(
                "The AI did not return valid JSON. Please try again."
            )

        quiz_data = json.loads(match.group(0))

    questions = quiz_data.get("questions", [])

    if len(questions) != number_of_questions:
        raise ValueError(
            f"Expected {number_of_questions} questions, "
            f"but received {len(questions)}. Generate another quiz."
        )

    # Validate each question before displaying it.
    for question in questions:
        if not all(
            key in question
            for key in ["question", "options", "correct_answer"]
        ):
            raise ValueError("A question is missing required fields.")

        if not isinstance(question["options"], dict):
            raise ValueError("Quiz options are in an invalid format.")

        if not all(
            option in question["options"]
            for option in ["A", "B", "C", "D"]
        ):
            raise ValueError("Each question must have four options.")

        correct_answer = str(question["correct_answer"]).strip().upper()

        if correct_answer not in ["A", "B", "C", "D"]:
            raise ValueError("A correct answer must be A, B, C, or D.")

        question["correct_answer"] = correct_answer
        question.setdefault("explanation", "No explanation provided.")

    return quiz_data


def save_quiz_attempt(user, topic, questions, answers, score):
    """Save the completed quiz and answers in MongoDB."""

    total_questions = len(questions)

    # Do not store correct answers in the displayed history document
    # unless needed for review; here they are retained for review.
    attempt = {
        "student": user,
        "topic": topic,
        "score": score,
        "total_questions": total_questions,
        "percentage": round(
            (score / total_questions) * 100, 2
        ) if total_questions else 0,
        "answers": [
            {
                "question": question["question"],
                "selected_answer": answers.get(str(index), ""),
                "correct_answer": question["correct_answer"],
                "is_correct": (
                    answers.get(str(index), "")
                    == question["correct_answer"]
                ),
                "explanation": question.get(
                    "explanation", "No explanation provided."
                )
            }
            for index, question in enumerate(questions)
        ],
        "created_at": datetime.now(timezone.utc)
    }

    result = quiz_collection.insert_one(attempt)
    return str(result.inserted_id)


# =========================================================
# 8. APP TABS
# =========================================================

chat_tab, quiz_tab, history_tab = st.tabs([
    "💬 Chat Tutor",
    "📝 Take a Quiz",
    "📊 My History"
])


# =========================================================
# 9. CHAT TUTOR PAGE
# =========================================================

with chat_tab:
    st.header("Ask your OS Tutor")

    st.write(
        "Ask questions about processes, deadlocks, memory management, "
        "CPU scheduling, file systems, and other OS concepts."
    )

    if st.button("Clear chat from this screen"):
        st.session_state["chat_screen_cleared"] = True
        st.rerun()

    # Persistent chat is retrieved from MongoDB.
    if not st.session_state.get("chat_screen_cleared", False):
        try:
            previous_messages = get_chat_history(username)
        except PyMongoError as error:
            st.error(f"Could not load chat history: {error}")
            previous_messages = []

        for message in previous_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    else:
        st.info(
            "The saved conversation remains in MongoDB. "
            "This button only hides the chat on this screen."
        )

    user_input = st.chat_input("Ask an Operating Systems question...")

    if user_input:
        st.session_state["chat_screen_cleared"] = False

        with st.chat_message("user"):
            st.write(user_input)

        try:
            save_chat_message(username, "user", user_input)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer, retrieved_documents = generate_chat_answer(
                        user_input
                    )
                st.write(answer)

                if retrieved_documents:
                    with st.expander("📚 View syllabus sources"):
                        for index, document in enumerate(
                            retrieved_documents, start=1
                        ):
                            st.markdown(f"**Source {index}**")
                            st.write(document.page_content[:500])

            save_chat_message(username, "assistant", answer)

        except Exception as error:
            st.error(f"Could not process the question: {error}")


# =========================================================
# 10. TAKE A QUIZ PAGE
# =========================================================

with quiz_tab:
    st.header("Test your Operating Systems knowledge")

    with st.form("quiz_generation_form"):
        topic = st.text_input(
            "Quiz topic",
            value="Deadlocks"
        ).strip()

        number_of_questions = st.selectbox(
            "Number of questions",
            options=[5, 10, 15],
            index=0
        )

        generate_button = st.form_submit_button("Generate New Quiz")

    if generate_button:
        if not topic:
            st.warning("Please enter a quiz topic.")
        else:
            try:
                with st.spinner("Generating your quiz..."):
                    new_quiz = generate_quiz(
                        topic,
                        number_of_questions
                    )

                st.session_state["active_quiz"] = new_quiz
                st.session_state["active_quiz_topic"] = topic
                st.session_state["active_quiz_submitted"] = False

                # Clear old answers when a new quiz is generated.
                st.session_state.pop("submitted_quiz_answers", None)

            except Exception as error:
                st.error(f"Could not generate the quiz: {error}")

    active_quiz = st.session_state.get("active_quiz")
    active_topic = st.session_state.get("active_quiz_topic", "OS")
    quiz_submitted = st.session_state.get(
        "active_quiz_submitted", False
    )

    if active_quiz:
        questions = active_quiz["questions"]

        if not quiz_submitted:
            st.subheader(f"Quiz: {active_topic}")
            st.write(f"Total questions: {len(questions)}")

            with st.form("answer_quiz_form"):
                selected_answers = {}

                for index, question in enumerate(questions):
                    st.markdown(
                        f"**Question {index + 1}: "
                        f"{question['question']}**"
                    )

                    options = question["options"]

                    selected_option = st.radio(
                        "Choose your answer:",
                        options=["A", "B", "C", "D"],
                        format_func=lambda option, opts=options: (
                            f"{option}. {opts[option]}"
                        ),
                        key=f"quiz_answer_{index}",
                        index=None
                    )

                    selected_answers[str(index)] = selected_option

                submit_quiz = st.form_submit_button("Submit Quiz")

            if submit_quiz:
                unanswered = [
                    index + 1
                    for index, question in enumerate(questions)
                    if selected_answers.get(str(index)) is None
                ]

                if unanswered:
                    st.warning(
                        "Please answer every question before submitting. "
                        f"Unanswered question numbers: {unanswered}"
                    )
                else:
                    score = sum(
                        selected_answers[str(index)]
                        == question["correct_answer"]
                        for index, question in enumerate(questions)
                    )

                    try:
                        attempt_id = save_quiz_attempt(
                            username,
                            active_topic,
                            questions,
                            selected_answers,
                            score
                        )

                        st.session_state["submitted_quiz_answers"] = (
                            selected_answers
                        )
                        st.session_state["active_quiz_score"] = score
                        st.session_state["active_quiz_submitted"] = True
                        st.session_state["active_quiz_attempt_id"] = (
                            attempt_id
                        )
                        st.rerun()

                    except Exception as error:
                        st.error(
                            "Your quiz was graded, but the attempt "
                            f"could not be saved: {error}"
                        )

        else:
            score = st.session_state["active_quiz_score"]
            total = len(questions)
            percentage = round((score / total) * 100, 2)

            st.subheader("Your quiz result")
            st.metric("Score", f"{score} / {total}")
            st.metric("Percentage", f"{percentage}%")

            if percentage >= 80:
                st.success("Excellent work!")
            elif percentage >= 50:
                st.info("Good effort! Review the incorrect answers.")
            else:
                st.warning("Keep practising. Review the explanations below.")

            submitted_answers = st.session_state[
                "submitted_quiz_answers"
            ]

            st.subheader("Answer review")

            for index, question in enumerate(questions):
                chosen = submitted_answers[str(index)]
                correct = question["correct_answer"]
                is_correct = chosen == correct

                with st.expander(
                    f"Question {index + 1}: "
                    f"{'Correct' if is_correct else 'Incorrect'}"
                ):
                    st.write(question["question"])
                    st.write(
                        f"Your answer: {chosen}. "
                        f"{question['options'][chosen]}"
                    )
                    st.write(
                        f"Correct answer: {correct}. "
                        f"{question['options'][correct]}"
                    )
                    st.write(
                        "Explanation: "
                        + question.get(
                            "explanation",
                            "No explanation provided."
                        )
                    )

            st.caption(
                "This attempt has been saved to MongoDB."
            )

            if st.button("Start another quiz"):
                st.session_state.pop("active_quiz", None)
                st.session_state.pop("active_quiz_topic", None)
                st.session_state.pop("active_quiz_submitted", None)
                st.session_state.pop("active_quiz_score", None)
                st.session_state.pop("submitted_quiz_answers", None)
                st.rerun()


# =========================================================
# 11. CHAT AND QUIZ HISTORY PAGE
# =========================================================

with history_tab:
    st.header("📊 My Learning History")

    history_chat_tab, history_quiz_tab = st.tabs([
        "Chat History",
        "Quiz History"
    ])

    # -----------------------------------------------------
    # Chat history
    # -----------------------------------------------------

    with history_chat_tab:
        st.subheader("Previous conversations")

        try:
            chat_records = list(
                chat_collection.find(
                    {"student": username}
                ).sort("created_at", -1).limit(200)
            )

            if not chat_records:
                st.info("No chat messages saved yet.")
            else:
                for record in chat_records:
                    role = record.get("role", "assistant")
                    content = record.get("content", "")
                    created_at = record.get("created_at")

                    with st.container(border=True):
                        st.caption(
                            f"{role.title()} • {created_at}"
                        )
                        st.write(content)

        except PyMongoError as error:
            st.error(f"Could not retrieve chat history: {error}")

    # -----------------------------------------------------
    # Quiz history
    # -----------------------------------------------------

    with history_quiz_tab:
        st.subheader("Previous quiz attempts")

        try:
            quiz_records = list(
                quiz_collection.find(
                    {"student": username}
                ).sort("created_at", -1).limit(100)
            )

            if not quiz_records:
                st.info("No completed quizzes saved yet.")
            else:
                total_attempts = len(quiz_records)
                total_score = sum(
                    record.get("score", 0)
                    for record in quiz_records
                )
                total_questions = sum(
                    record.get("total_questions", 0)
                    for record in quiz_records
                )

                average_percentage = (
                    round(total_score / total_questions * 100, 2)
                    if total_questions else 0
                )

                col1, col2, col3 = st.columns(3)

                col1.metric("Quiz attempts", total_attempts)
                col2.metric("Questions answered", total_questions)
                col3.metric("Overall accuracy", f"{average_percentage}%")

                for record in quiz_records:
                    score = record.get("score", 0)
                    total = record.get("total_questions", 0)
                    percentage = record.get("percentage", 0)
                    created_at = record.get("created_at")
                    quiz_topic = record.get("topic", "OS")

                    with st.expander(
                        f"{quiz_topic} — {score}/{total} "
                        f"({percentage}%)"
                    ):
                        st.write(f"Date: {created_at}")
                        st.write(f"Score: {score} out of {total}")
                        st.write(f"Percentage: {percentage}%")

                        for index, answer in enumerate(
                            record.get("answers", []), start=1
                        ):
                            st.markdown(
                                f"**Question {index}:** "
                                f"{answer.get('question', '')}"
                            )

                            selected = answer.get(
                                "selected_answer", ""
                            )
                            correct = answer.get(
                                "correct_answer", ""
                            )

                            st.write(f"Your answer: {selected}")
                            st.write(f"Correct answer: {correct}")

                            st.write(
                                answer.get("explanation", "")
                            )

        except PyMongoError as error:
            st.error(f"Could not retrieve quiz history: {error}")


# =========================================================
# 12. FOOTER
# =========================================================

st.sidebar.divider()
st.sidebar.caption("OSTutorLLM • Learn OS step by step")