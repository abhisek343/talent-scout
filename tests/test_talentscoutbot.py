import pytest
from unittest.mock import patch, MagicMock
from talent_scout_app import TalentScoutBot

@pytest.fixture
def mock_bot():
    # Create the bot object, but mock out external dependencies
    with patch("talent_scout_app.create_connection", return_value=MagicMock()):
        with patch("talent_scout_app.create_table", return_value=True):
            bot = TalentScoutBot()
    # Now override any methods that hit external services (OpenAI)
    bot.conversation.predict = MagicMock(return_value="Dummy AI response")
    return bot

def test_bot_initialization(mock_bot):
    # If we got here, it means the DB connection and table creation were "successful"
    assert mock_bot.db_conn is not None
    assert mock_bot.memory is not None

def test_generate_bot_response_welcome(mock_bot):
    # For a stage not expecting a tuple
    response = mock_bot.generate_bot_response("Hello", "welcome", {})
    assert isinstance(response, str)
    assert "Dummy AI response" in response

def test_generate_bot_response_tech_stack(mock_bot):
    # For a stage expecting a tuple
    mock_bot._extract_tech_stack = MagicMock(return_value=["python", "java"])
    response, tech_stack = mock_bot.generate_bot_response("I know Python and Java", "tech_stack", {})
    assert isinstance(response, str)
    assert tech_stack == ["python", "java"]

def test_generate_bot_response_technical_questions(mock_bot):
    mock_bot.current_question_idx = 1
    mock_bot.technical_questions = [{"question": "Q1"}]
    mock_bot._evaluate_technical_answer = MagicMock(return_value={"score": 9, "feedback": "Excellent"})
    response, evaluation = mock_bot.generate_bot_response("My answer", "technical_questions", {"tech_stack": ["python"]})
    assert isinstance(response, str)
    assert evaluation["score"] == 9
    assert "Excellent" in evaluation["feedback"]
