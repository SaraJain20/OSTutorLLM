
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime, timezone

# Load environment variables from .env
load_dotenv()

# Read the MongoDB connection string
uri = os.getenv("MONGODB_URI")

if not uri:
    raise ValueError("MONGODB_URI is missing from your .env file")

client = MongoClient(uri, serverSelectionTimeoutMS=10000)

try:
    # Test the connection
    client.admin.command("ping")
    print("SUCCESS: Connected to MongoDB Atlas!")

    # Select the database and collection
    db = client["ostutor_db"]
    collection = db["quiz_attempts"]

    # Insert one test record
    test_record = {
        "student": "test_student",
        "topic": "Operating Systems",
        "score": 4,
        "total_questions": 5,
        "created_at": datetime.now(timezone.utc)
    }

    result = collection.insert_one(test_record)
    print("Inserted document ID:", result.inserted_id)

    # Read the record back
    saved_record = collection.find_one({"_id": result.inserted_id})
    print("Record retrieved successfully:")
    print(saved_record)

except Exception as e:
    print("MongoDB test failed:", type(e).__name__, str(e))

finally:
    client.close()
