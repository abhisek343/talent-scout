import pytest
from talent_scout_app import (
    validate_phone,
    validate_email_address,
    validate_experience,
    get_candidate_embeddings,
)

@pytest.mark.parametrize(
    "phone_input, expected_valid, expected_start",
    [
        ("+919876543210", True, "+91"),
        ("+123", False, None),            # Too short to be valid
        ("9876543210", False, None),      # Missing '+' prefix
    ]
)
def test_validate_phone(phone_input, expected_valid, expected_start):
    is_valid, result = validate_phone(phone_input)
    assert is_valid == expected_valid
    if expected_valid:
        # result is the formatted phone number
        assert result.startswith(expected_start)
    else:
        # result is the error message
        assert isinstance(result, str)

@pytest.mark.parametrize(
    "email_input, expected_valid",
    [
        ("test@example.com", True),
        ("invalid-email", False),
        ("another_test@domain.co.in", True),
    ]
)
def test_validate_email_address(email_input, expected_valid):
    is_valid, result = validate_email_address(email_input)
    assert is_valid == expected_valid

@pytest.mark.parametrize(
    "experience_input, expected_valid, expected_value",
    [
        ("5", True, 5.0),
        ("0", True, 0.0),
        ("-3", False, None),
        ("abc", False, None),
        ("51", False, None),
    ]
)
def test_validate_experience(experience_input, expected_valid, expected_value):
    is_valid, value, error = validate_experience(experience_input)
    assert is_valid == expected_valid
    if expected_valid:
        assert value == expected_value
    else:
        assert isinstance(error, str)
        assert value == 0 or value == 0.0

def test_get_candidate_embeddings():
    embeddings = get_candidate_embeddings(["python", "javascript"], "Software Engineer")
    assert isinstance(embeddings, list)
    assert len(embeddings) == 10
    for emb in embeddings:
        assert isinstance(emb, float)
