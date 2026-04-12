import chromadb
import os
import google.generativeai as genai

DATABASE_FILE_PATH = "./chroma_db_data"

exampleSourceDocuments = [
    "Course registration for autumn semester opens on 15 August and closes on 31 August.",
    "Students can reset their university password through the self-service portal at password.uni.fi.",
    "The library is open from Monday to Friday until 20:00 during teaching periods.",
    "the library is open from Monday to Friday until 18:00 during non-teaching periods.",
    "Exam re-sit registration must be completed at least 7 days before the exam date.",
    "Student email accounts use the format firstname.lastname@student.uni.fi.",
    "The IT helpdesk is located in the main building, room A120, and is open from 9:00 to 15:00.",
    "Eduroam Wi-Fi is available on campus and students should log in using their full student email address.",
    "Printing costs 0.04 EUR per black-and-white page and 0.15 EUR per color page."
]

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set. Please set it.")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash-lite")


def initVectorDb():
    print("Initializing vector database...")

    client = chromadb.PersistentClient(path=DATABASE_FILE_PATH)

    try:
        client.delete_collection(name="university_helpdesk_knowledge_base")
    except:
        pass

    collection = client.get_or_create_collection(name="university_helpdesk_knowledge_base")

    collection.add(
        documents=exampleSourceDocuments,
        ids=[str(i) for i in range(len(exampleSourceDocuments))]
    )

    print(f"Inserted {len(exampleSourceDocuments)} documents into the collection.")
    print("Vector database initialized successfully.")


def query_gemini(prompt):
    response = model.generate_content(prompt)
    return response.text.strip()


def queryVectorDb(query, n_results=5):
    print(f"Querying vector database for: {query}")

    client = chromadb.PersistentClient(path=DATABASE_FILE_PATH)
    collection = client.get_collection(name="university_helpdesk_knowledge_base")

    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    print(f"Found {len(results['documents'][0])} results for the query.")
    print("\nRetrieved chunks:")
    for i, doc in enumerate(results["documents"][0], start=1):
        print(f"{i}. {doc}")

    return results


def create_context_from_vector_db(question, n_results=5):
    results = queryVectorDb(question, n_results=n_results)
    return " ".join(results["documents"][0])


def rag_query(question, n_results=5):
    context = create_context_from_vector_db(question, n_results=n_results)
    prompt = f"""
You are a helpful FAQ assistant.
Answer only using the provided context.
If the answer is not in the context, say: "I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""
    return query_gemini(prompt)


def query_without_rag(question):
    prompt = f"""
Question:
{question}

Answer:
"""
    return query_gemini(prompt)


initVectorDb()

search_query = "When does the library close?"
# search_query = "How do I reset my password?"
# search_query = "How much does color printing cost?"
#search_query = "Where is the IT helpdesk?"
# = "When can I register for autumn semester courses?"
#search_query = "cost of prints"
# search_query = "What time does the gym close?"
#search_query = "How to reset my pasvord"
#search_query = "How the student emails are created?"
#search_query = "How many students are in the university?"


RAGresults = rag_query(search_query, n_results=10)
print("RAG Results\n###########################################################")
print(f"Query: {search_query}")
print(f"LLM Response: {RAGresults}\n")

print("************************************************************")

NoContextResults = query_without_rag(search_query)
print("No RAG Results\n###########################################################")
print(f"Query: {search_query}")
print(f"LLM Response: {NoContextResults}\n")