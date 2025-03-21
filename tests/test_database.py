import sys
import os

# Add the "talent-scout" folder (project root for modules) into sys.path.
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "talent-scout"))
sys.path.insert(0, module_path)

import pytest
import psycopg2
from unittest.mock import patch, MagicMock
from database import create_connection, create_table, insert_candidate, DuplicateCandidateException

@pytest.fixture
def mock_conn():
    # Create a mock psycopg2 connection
    conn = MagicMock()
    conn.cursor.return_value = MagicMock()
    return conn

def test_create_table_success(mock_conn):
    # Patch create_connection to return our mock_conn
    with patch("database.create_connection", return_value=mock_conn):
        result = create_table(mock_conn)
        assert result is True
        mock_conn.cursor.assert_called_once()
        mock_conn.commit.assert_called_once()

def test_insert_candidate_success(mock_conn):
    candidate_info = {
        "email": "test@example.com",
        "name": "John Doe",
        "phone": "+919876543210",
        "experience": 3.0,
        "location": "Mumbai",
        "position": "Software Engineer",
        "tech_stack": ["python", "aws"],
        "technical_answers": {"Q1": {"score": 7, "feedback": "Good"}},
        "evaluation": {"overall_score": 8.0},
        "embeddings": [0.1, 0.2],
        "timestamp": None
    }
    with patch("database.create_connection", return_value=mock_conn):
        result = insert_candidate(mock_conn, candidate_info)
        assert result == "Candidate inserted successfully"
        mock_conn.cursor.return_value.execute.assert_called_once()
        mock_conn.commit.assert_called_once()

def test_insert_candidate_duplicate_error(mock_conn):
    # Use a MagicMock with spec=psycopg2.Error to simulate a unique violation.
    mock_cursor = mock_conn.cursor.return_value
    error = MagicMock(spec=psycopg2.Error)
    error.pgcode = "23505"  # Unique violation code
    mock_cursor.execute.side_effect = error

    candidate_info = {
        "email": "duplicate@example.com",
        "name": "Jane",
        "phone": "+919876543210",
        "experience": 2.0,
        "location": "Delhi",
        "position": "DevOps Engineer",
        "tech_stack": ["docker", "kubernetes"],
        "technical_answers": {},
        "evaluation": {},
        "embeddings": [],
        "timestamp": None
    }
    with patch("database.create_connection", return_value=mock_conn):
        with pytest.raises(DuplicateCandidateException):
            insert_candidate(mock_conn, candidate_info)
