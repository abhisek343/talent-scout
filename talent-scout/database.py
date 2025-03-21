import psycopg2
import os
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Custom exception for duplicate candidate; subclassing psycopg2.Error so tests expecting a psycopg2 error pass.
class DuplicateCandidateException(psycopg2.Error):
    pass

def create_connection():
    try:
        dbname = os.getenv("DB_NAME")
        user = os.getenv("DB_USER")
        password = os.getenv("DB_PASSWORD")
        host = os.getenv("DB_HOST")
        port = os.getenv("DB_PORT")
        
        missing_vars = []
        if not dbname: missing_vars.append("DB_NAME")
        if not user: missing_vars.append("DB_USER")
        if not password: missing_vars.append("DB_PASSWORD")
        if not host: missing_vars.append("DB_HOST")
        if not port: missing_vars.append("DB_PORT")
        
        if missing_vars:
            print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
            return None
            
        conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )
        print("✅ Database connection successful!")
        return conn
    except psycopg2.Error as e:
        print(f"❌ Database connection failed: {e}")
        if "password authentication failed" in str(e).lower():
            print("💡 Hint: Check if the DB_PASSWORD environment variable is correct")
        elif "could not translate host name" in str(e).lower() or "could not connect to server" in str(e).lower():
            print("💡 Hint: Check if the DB_HOST is correct and accessible")
        return None

def create_table(conn):
    try:
        if conn is None:
            print("❌ Cannot create table: No database connection")
            return False
        
        cur = conn.cursor()
        # For testing, drop the table first to ensure the updated schema is used.
        if os.getenv("TESTING", "false").lower() == "true":
            cur.execute("DROP TABLE IF EXISTS candidates;")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS candidates (
                email TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                phone TEXT NOT NULL,
                experience FLOAT NOT NULL,
                location TEXT NOT NULL,
                position TEXT NOT NULL,
                tech_stack TEXT NOT NULL,
                technical_answers JSONB,
                evaluation JSONB,
                embeddings JSONB,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.commit()
        cur.close()
        print("✅ Candidates table is ready!")
        return True
    except psycopg2.Error as e:
        print(f"❌ Error creating table: {e}")
        return False

def insert_candidate(conn, candidate_info):
    """Insert candidate data into the PostgreSQL candidates table."""
    try:
        if conn is None:
            print("❌ Cannot insert candidate: No database connection")
            return "Database connection error"
            
        if "timestamp" not in candidate_info:
            candidate_info["timestamp"] = datetime.now().isoformat()
        
        insert_query = """
        INSERT INTO candidates (email, name, phone, experience, location, position, tech_stack, 
                               technical_answers, evaluation, embeddings, timestamp)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cur = conn.cursor()

        tech_stack = candidate_info.get("tech_stack", [])
        if not isinstance(tech_stack, list):
            tech_stack = [tech_stack]
        tech_stack_str = ", ".join(str(tech) for tech in tech_stack)
        
        technical_answers = candidate_info.get("technical_answers", {})
        technical_answers_json = json.dumps(technical_answers)
        
        evaluation = candidate_info.get("evaluation", {})
        evaluation_json = json.dumps(evaluation)
        
        embeddings = candidate_info.get("embeddings", [])
        embeddings_json = json.dumps(embeddings)

        timestamp = candidate_info.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        cur.execute(insert_query, (
            candidate_info.get("email"),
            candidate_info.get("name"),
            candidate_info.get("phone"),
            candidate_info.get("experience"),
            candidate_info.get("location"),
            candidate_info.get("position"),
            tech_stack_str,
            technical_answers_json,
            evaluation_json,
            embeddings_json,
            timestamp
        ))
        conn.commit()
        cur.close()
        print("✅ Candidate inserted successfully!")
        return "Candidate inserted successfully"
    except psycopg2.Error as e:
        if e.pgcode == "23505":  # Unique violation for duplicate email
            print("⚠️ Candidate already exists with this email")
            raise DuplicateCandidateException("Candidate already exists with this email")
        else:
            print(f"❌ Error inserting candidate: {e}")
            return f"Error inserting candidate: {e}"
