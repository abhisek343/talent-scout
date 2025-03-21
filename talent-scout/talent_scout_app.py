import os
import streamlit as st
import logging
import json
import uuid
from datetime import datetime
import time
from typing import Dict, List, Any, Tuple, Union
from pathlib import Path

# Advanced imports
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from email_validator import validate_email, EmailNotValidError
import phonenumbers
import pycountry
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv, find_dotenv

# AI/ML imports
import openai
# Alias for OpenAI so tests can patch talent_scout_app.OpenAI
OpenAI = openai
import numpy as np
from database import create_connection, create_table, insert_candidate, DuplicateCandidateException

# Consolidated UI components import (with fallback definitions)
try:
    from custom_ui import load_custom_css, load_advanced_ui, st_phone_number
except ImportError:
    def load_custom_css():
        pass
    def load_advanced_ui():
        pass
    def st_phone_number(label, placeholder="", default_country="CN"):
         return st.text_input(label, placeholder=placeholder)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path)
    logger.info(f"Loaded .env file from {dotenv_path}")
else:
    logger.warning("No .env file found. Using environment variables from system.")

# Check required environment variables
required_env_vars = ["DB_NAME", "DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT", "OPENAI_API_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Helper Functions ---
def validate_phone(phone: str) -> Tuple[bool, str]:
    """
    Validate and format a phone number.
    Candidate must input the number in international format.
    For example, for India: +91 followed by 10 digits.
    Returns:
        (True, formatted_number) if valid, or (False, error_message) if not.
    """
    if not phone.strip().startswith("+"):
        return False, "Please input your phone number in the correct format: for India, use +91 followed by 10 digits."
    try:
        parsed_number = phonenumbers.parse(phone, None)
        if phonenumbers.is_possible_number(parsed_number) and phonenumbers.is_valid_number(parsed_number):
            formatted_number = phonenumbers.format_number(parsed_number, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
            return True, formatted_number
        else:
            return False, "Invalid phone number"
    except Exception as e:
        return False, str(e)

def validate_experience(experience: str) -> Tuple[bool, float, str]:
    try:
        clean_exp = ''.join(c for c in experience if c.isdigit() or c in ".-")
        exp_value = float(clean_exp)
        if exp_value < 0:
            return False, 0, "Experience cannot be negative"
        if exp_value > 50:
            return False, 0, "Please enter a realistic experience value (0-50 years)"
        return True, exp_value, ""
    except ValueError:
        return False, 0, "Please enter a valid number for years of experience"

def validate_email_address(email: str) -> Tuple[bool, str]:
    try:
        validation = validate_email(email, check_deliverability=False)
        return True, validation.normalized
    except EmailNotValidError as e:
        return False, str(e)

def get_candidate_embeddings(tech_stack: List[str], position: str) -> List[float]:
    try:
        combined_text = f"{position} {' '.join(tech_stack)}"
        embeddings = list(np.random.rand(10).astype(float))
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return [0.0] * 10

# --- Subclass to Allow Patching of ConversationChain ---
class MutableConversationChain(ConversationChain):
    class Config:
        allow_mutation = True

# --- TalentScoutBot Class ---
class TalentScoutBot:
    """Advanced TalentScout chatbot with enhanced AI capabilities."""
    
    STAGES = [
        "welcome", "email", "phone", "experience", "position",
        "location", "tech_stack", "technical_questions", "evaluation", "end"
    ]
    
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.error("OpenAI API Key not found in environment variables.")
            raise ValueError("❌ OpenAI API Key not found in environment variables.")
        openai.api_key = self.openai_api_key
        try:
            self.chat_model = ChatOpenAI(
                model="gpt-4",
                temperature=0.7,
                max_tokens=1500,
                api_key=self.openai_api_key
            )
            self.memory = ConversationBufferMemory(return_messages=True)
            self.conversation = MutableConversationChain(llm=self.chat_model, memory=self.memory, verbose=True)
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            raise ValueError(f"❌ Could not initialize OpenAI client: {e}")
        
        self.db_conn = self._init_db_connection_with_retry()
        if self.db_conn is None:
            logger.error("Could not connect to the PostgreSQL database.")
            raise ConnectionError("❌ Could not connect to the PostgreSQL database.")
        try:
            create_table(self.db_conn)
        except Exception as e:
            logger.error(f"Error creating table: {e}")
            raise e
        self.country_codes = self._get_country_codes()
        self.tech_question_cache = {}
        self.current_question_idx = 0
        self.technical_questions = []
        self.asked_questions = set()
    
    def _init_db_connection_with_retry(self, max_attempts=3):
        for attempt in range(max_attempts):
            try:
                conn = create_connection()
                if conn is not None:
                    return conn
                logger.warning(f"Database connection attempt {attempt+1}/{max_attempts} failed. Retrying...")
                time.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Error connecting to database (attempt {attempt+1}/{max_attempts}): {e}")
                if attempt == max_attempts - 1:
                    return None
                time.sleep(2 ** attempt)
        return None
    
    def _get_country_codes(self) -> Dict[str, str]:
        country_codes = {}
        for country in pycountry.countries:
            try:
                country_code = phonenumbers.country_code_for_region(country.alpha_2)
                flag = self._get_flag_emoji(country.alpha_2)
                country_codes[f"{flag} {country.name} (+{country_code})"] = f"+{country_code}"
            except Exception as e:
                logger.debug(f"Could not get country code for {country.name}: {e}")
                continue
        return country_codes
    
    def _get_flag_emoji(self, country_code: str) -> str:
        if len(country_code) != 2:
            return "🌍"
        return "".join(chr(ord(c.upper()) + 127397) for c in country_code)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_bot_response(self, user_input: str, stage: str, candidate_info: Dict[str, Any]) -> Union[str, Tuple[str, Any]]:
        try:
            prompt = self._create_stage_prompt(stage, candidate_info, user_input)
            response = self.conversation.predict(input=prompt)
            if stage == "tech_stack":
                tech_stack = self._extract_tech_stack(user_input, response)
                return response, tech_stack
            elif stage == "technical_questions":
                if self.current_question_idx > 0 and self.technical_questions:
                    try:
                        evaluation = self._evaluate_technical_answer(
                            question=self.technical_questions[self.current_question_idx-1]["question"],
                            answer=user_input,
                            tech_stack=candidate_info["tech_stack"]
                        )
                    except Exception as eval_e:
                        logger.error(f"Error evaluating technical answer: {eval_e}")
                        evaluation = {"score": 5, "feedback": "Evaluation failed. A human reviewer will assess it."}
                    return response, evaluation
                else:
                    logger.warning(f"Invalid question index: {self.current_question_idx}")
                    return response, {"score": 5, "feedback": "Could not evaluate answer properly."}
            return response
        except Exception as e:
            logger.error(f"Error generating bot response: {e}")
            if stage == "tech_stack":
                return "I apologize, but I'm having trouble processing your tech stack. Please try again.", []
            elif stage == "technical_questions":
                return "I apologize, but I'm having trouble processing your technical answer. Please try again.", {"score": 5, "feedback": "Evaluation failed."}
            else:
                return "I apologize, but I'm having trouble processing your response. Let's try again."
    
    def _create_stage_prompt(self, stage: str, candidate_info: Dict[str, Any], user_input: str) -> str:
        base_prompt = f"You are TalentScout, a professional AI hiring assistant. You should respond in a direct, professional manner without using phrases like 'Dear candidate', 'Sure', or 'Absolutely'. The candidate has provided: {user_input}. "
        stage_prompts = {
            "welcome": "Thank the candidate for providing their name and ask for their email address.",
            "email": "The candidate provided their email. Ask for their contact phone number.",
            "phone": "The candidate provided their phone number. Validate it and then ask for their years of experience.",
            "experience": "The candidate provided their experience. Ask what position they're applying for.",
            "position": "The candidate provided their position. Ask for their current location.",
            "location": "The candidate provided their location. Ask about their technical skills and tech stack.",
            "tech_stack": f"The candidate mentioned these technologies: {user_input}. Extract the tech stack, thank them professionally, and prepare for technical questions.",
            "technical_questions": "The candidate has answered a technical question. Provide professional feedback and move to the next question.",
            "evaluation": "Summarize the candidate's performance and provide next steps.",
            "end": "Thank the candidate for their time and inform them that their application has been received."
        }
        return base_prompt + stage_prompts.get(stage, "Continue the conversation in a professional tone.")
    
    def _extract_tech_stack(self, user_input: str, ai_response: str) -> List[str]:
        try:
            extraction_prompt = f"""
            Extract all technologies and programming languages from this text:
            "{user_input}"
            
            Format the result as a JSON array of strings with only the names of the technologies.
            """
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": extraction_prompt}],
                temperature=0.1,
                max_tokens=200
            )
            tech_stack_text = response.choices[0].message.content
            try:
                tech_stack = json.loads(tech_stack_text)
                if not isinstance(tech_stack, list):
                    tech_stack = [tech_stack]
                return tech_stack
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tech stack JSON: {tech_stack_text}")
                words = user_input.lower().split()
                common_tech = ["python", "javascript", "java", "c++", "react", "node", "sql", "aws"]
                return [word for word in words if word.lower() in common_tech]
        except Exception as e:
            logger.error(f"Error extracting tech stack: {e}")
            words = user_input.split()
            common_tech = ["python", "javascript", "java", "c++", "react", "node", "sql", "aws"]
            return [word for word in words if word.lower() in common_tech]
    
    def _evaluate_technical_answer(self, question: str, answer: str, tech_stack: List[str]) -> Dict[str, Any]:
        try:
            evaluation_prompt = f"""
            Evaluate this technical answer for a {', '.join(tech_stack)} position:
            
            Question: {question}
            Answer: {answer}
            
            Provide a JSON response with the following fields:
            - score: A number from 1-10
            - feedback: Constructive feedback about the answer
            - strengths: What was good about the answer
            - areas_to_improve: What could be improved
            - accurate: Boolean indicating if the answer is technically accurate
            """
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": evaluation_prompt}],
                temperature=0.2,
                max_tokens=500
            )
            evaluation_text = response.choices[0].message.content
            try:
                evaluation = json.loads(evaluation_text)
                return evaluation
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse evaluation JSON: {evaluation_text}")
                return {
                    "score": 5,
                    "feedback": "We had trouble evaluating your answer automatically. A human reviewer will assess it.",
                    "strengths": "N/A",
                    "areas_to_improve": "N/A",
                    "accurate": None
                }
        except Exception as e:
            logger.error(f"Error evaluating technical answer: {e}")
            return {
                "score": 5,
                "feedback": "We had trouble evaluating your answer automatically. A human reviewer will assess it.",
                "strengths": "N/A",
                "areas_to_improve": "N/A",
                "accurate": None
            }
    
    def generate_technical_questions(self, tech_stack: List[str], experience_level: float) -> List[Dict[str, Any]]:
        if not tech_stack:
            tech_stack = ["general programming"]
        cache_key = f"{'-'.join(sorted(tech_stack))}-{experience_level}"
        if cache_key in self.tech_question_cache:
            return self.tech_question_cache[cache_key]
        if experience_level < 2:
            difficulty = "basic"
        elif experience_level < 5:
            difficulty = "intermediate"
        else:
            difficulty = "advanced"
        try:
            prompt = f"""
            Generate 5 {difficulty} technical questions for a candidate with {experience_level} years of experience 
            in {', '.join(tech_stack)}. 
            
            For each question, provide:
            1. The question text
            2. Expected answer points
            3. Technology category
            
            Format the output as a JSON array of objects with the following structure:
            [
                {{
                    "question": "Question text",
                    "expected_answer": "Key points that should be in the answer",
                    "technology": "Specific technology this tests",
                    "difficulty": "{difficulty}"
                }}
            ]
            """
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1500
            )
            questions_text = response.choices[0].message.content
            try:
                questions = json.loads(questions_text)
                if len(questions) < 5:
                    default_qs = self._generate_default_questions(tech_stack, difficulty)
                    questions.extend(default_qs[len(questions):])
                self.tech_question_cache[cache_key] = questions
                return questions
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse technical questions JSON: {questions_text}")
                return self._generate_default_questions(tech_stack, difficulty)
        except Exception as e:
            logger.error(f"Error generating technical questions: {e}")
            return self._generate_default_questions(tech_stack, difficulty)

    def _generate_default_questions(self, tech_stack: List[str], difficulty: str) -> List[Dict[str, Any]]:
        if not tech_stack:
            tech_stack = ["general programming"]
        return [
            {
                "question": f"Explain your experience with {tech}" if tech else "Describe your technical background",
                "expected_answer": "Candidate should demonstrate knowledge and experience",
                "technology": tech if tech else "general",
                "difficulty": difficulty
            }
            for tech in (tech_stack[:5] if len(tech_stack) >= 5 else tech_stack + ["general"] * (5 - len(tech_stack)))
        ]
    
    def calculate_candidate_score(self, candidate_info: Dict[str, Any]) -> Dict[str, Any]:
        try:
            tech_answers = candidate_info.get("technical_answers", {})
            if not tech_answers:
                return {
                    "technical_score": 5.0,
                    "experience_score": 5.0,
                    "overall_score": 5.0,
                    "feedback": "No technical questions were answered. Further evaluation required.",
                    "recommendation": "Review Required"
                }
            scores = [answer.get("score", 5) for answer in tech_answers.values()]
            avg_score = sum(scores) / len(scores) if scores else 5
            exp_score = min(candidate_info.get("experience", 0) * 2, 10)
            overall_score = (avg_score * 0.7) + (exp_score * 0.3)
            scoring_prompt = f"""
            Generate a brief feedback summary for a candidate with the following profile:
            - Position: {candidate_info.get('position', 'Software Developer')}
            - Experience: {candidate_info.get('experience', 0)} years
            - Tech Stack: {', '.join(candidate_info.get('tech_stack', []))}
            - Technical Interview Score: {avg_score}/10
            
            Provide a brief assessment of their strengths and areas for improvement.
            """
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": scoring_prompt}],
                    temperature=0.7,
                    max_tokens=500
                )
                feedback = response.choices[0].message.content
            except Exception as e:
                logger.error(f"Error generating feedback: {e}")
                feedback = "Automatic feedback generation failed. A human recruiter will review the application."
            return {
                "technical_score": round(avg_score, 1),
                "experience_score": round(exp_score, 1),
                "overall_score": round(overall_score, 1),
                "feedback": feedback,
                "recommendation": "Strong Match" if overall_score >= 8 else 
                                  "Potential Match" if overall_score >= 6 else 
                                  "Not Recommended"
            }
        except Exception as e:
            logger.error(f"Error calculating candidate score: {e}")
            return {
                "technical_score": 5.0,
                "experience_score": 5.0,
                "overall_score": 5.0,
                "feedback": "We had trouble generating automatic feedback. A human recruiter will review your application.",
                "recommendation": "Review Required"
            }
    
    def save_candidate_data(self, candidate_info: Dict[str, Any]) -> bool:
        if not all(key in candidate_info for key in ["name", "email", "phone", "experience", "position", "location"]):
            logger.error("Missing required candidate information")
            raise ValueError("Missing required candidate information")
        try:
            if "tech_stack" not in candidate_info or not candidate_info["tech_stack"]:
                candidate_info["tech_stack"] = ["Not specified"]
            tech_stack = candidate_info.get("tech_stack", [])
            position = candidate_info.get("position", "")
            candidate_info["embeddings"] = get_candidate_embeddings(tech_stack, position)
            candidate_info["evaluation"] = self.calculate_candidate_score(candidate_info)
            insert_candidate(self.db_conn, candidate_info)
            return True
        except DuplicateCandidateException as e:
            logger.warning(f"Duplicate candidate: {e}")
            raise e
        except Exception as e:
            logger.error(f"Error saving candidate data: {e}")
            raise e

# --- Main Streamlit UI Code ---
def main():
    st.set_page_config(
        page_title="TalentScout - AI Hiring Assistant",
        page_icon="👔",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load UI styles
    load_custom_css()
    load_advanced_ui()
    
    with st.sidebar:
        st.title("TalentScout")
        st.subheader("AI Hiring Assistant")
        st.write("---")
        st.write("This application helps screen candidates through an interactive chat interface.")
        if st.button("Reset Conversation"):
            st.session_state.clear()
            st.experimental_rerun()
        st.write("### System Status")
        try:
            db_conn = create_connection()
            if db_conn:
                st.success("✅ System Online")
            else:
                st.error("❌ Database Connection Failed")
                st.info("Check your .env file and database settings")
        except Exception as e:
            st.error(f"❌ Database Error: {str(e)}")
        if "current_stage" in st.session_state and "bot" in st.session_state:
            current_stage = st.session_state.current_stage
            progress = (TalentScoutBot.STAGES.index(current_stage) + 1) / len(TalentScoutBot.STAGES)
            st.write("### Interview Progress")
            st.progress(progress)
    
    st.title("TalentScout: AI Hiring Assistant")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "bot" not in st.session_state:
        if not os.getenv("OPENAI_API_KEY"):
            st.warning("⚠️ OpenAI API Key not found. Please set it in your environment variables to continue.")
            st.info("Ensure your .env file is in the correct location and properly formatted.")
            st.stop()
        try:
            db_conn = create_connection()
            if not db_conn:
                st.error("❌ Database connection failed. Please check your .env file and ensure your PostgreSQL server is running.")
                st.stop()
        except Exception as e:
            st.error(f"❌ Database connection error: {str(e)}")
            st.stop()
        try:
            st.session_state.bot = TalentScoutBot()
            welcome_msg = "👋 Welcome to the TalentScout AI Hiring Assistant! I'll help you through the application process. Let's start with your full name."
            st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
            st.session_state.current_stage = "welcome"
            st.session_state.candidate_info = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "technical_answers": {}
            }
        except Exception as e:
            st.error(f"Error initializing TalentScout: {str(e)}")
            st.stop()
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # If the interview has ended, show the final message and close input
    if st.session_state.get("current_stage", "") == "end":
        st.markdown("## Interview Concluded")
        st.write("Thank you for answering all technical questions. We are now concluding your interview.")
        st.stop()
    
    # Otherwise, allow chat input
    prompt = st.chat_input("Your response...")
    if prompt is not None:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("⏳ Thinking...")
            try:
                stage = st.session_state.current_stage
                valid_input = True
                error_message = ""
                
                if stage == "welcome":
                    if len(prompt.strip()) < 3:
                        valid_input = False
                        error_message = "Please provide your full name."
                    else:
                        st.session_state.candidate_info["name"] = prompt.strip()
                        response = f"Thank you, {prompt.strip()}. Please provide your email address."
                        message_placeholder.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.session_state.current_stage = "email"
                        valid_input = False
                        return
                elif stage == "email":
                    is_valid, email = validate_email_address(prompt)
                    if not is_valid:
                        valid_input = False
                        error_message = f"Invalid email address: {email}. Please provide a valid email (e.g., example@domain.com)."
                        message_placeholder.markdown(f"❌ {error_message}")
                        return
                    else:
                        st.session_state.candidate_info["email"] = email
                        response = "Could you please provide your contact phone number? (Format example: +91XXXXXXXXXX)"
                        message_placeholder.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.session_state.current_stage = "phone"
                        valid_input = False
                        return
                elif stage == "phone":
                    is_valid, formatted_phone = validate_phone(prompt)
                    if not is_valid:
                        valid_input = False
                        error_message = f"Invalid phone number: {formatted_phone}"
                    else:
                        st.session_state.candidate_info["phone"] = formatted_phone
                        response = f"Thank you, your phone number {formatted_phone} has been recorded. Can you please share how many years of experience you have in your field?"
                        message_placeholder.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.session_state.current_stage = "experience"
                        valid_input = False
                        return
                elif stage == "experience":
                    is_valid, exp_value, exp_error = validate_experience(prompt)
                    if not is_valid:
                        valid_input = False
                        error_message = exp_error
                    else:
                        st.session_state.candidate_info["experience"] = exp_value
                elif stage == "position":
                    if len(prompt.strip()) < 3:
                        valid_input = False
                        error_message = "Please provide a valid position."
                    else:
                        st.session_state.candidate_info["position"] = prompt.strip()
                elif stage == "location":
                    if len(prompt.strip()) < 3:
                        valid_input = False
                        error_message = "Please provide a valid location."
                    else:
                        st.session_state.candidate_info["location"] = prompt.strip()
                
                if not valid_input:
                    if error_message:
                        message_placeholder.markdown(f"❌ {error_message}")
                    return
                
                bot = st.session_state.bot
                result = bot.generate_bot_response(prompt, stage, st.session_state.candidate_info)
                
                if stage == "tech_stack":
                    response, tech_stack = result
                    st.session_state.candidate_info["tech_stack"] = tech_stack
                    experience = st.session_state.candidate_info.get("experience", 0)
                    st.session_state.bot.technical_questions = bot.generate_technical_questions(tech_stack, experience)
                    st.session_state.bot.current_question_idx = 0
                    st.session_state.bot.asked_questions = set()
                    
                    if st.session_state.bot.technical_questions:
                        first_q = st.session_state.bot.technical_questions[0]["question"]
                        response += f"\n\nLet's start with a technical question:\n\nQuestion 1: {first_q}"
                        st.session_state.bot.current_question_idx = 1
                        st.session_state.bot.asked_questions.add(first_q)
                        st.session_state.current_stage = "technical_questions"
                    else:
                        response += "\n\nI apologize, but I'm having trouble generating technical questions. Let's proceed with concluding the interview."
                        try:
                            bot.save_candidate_data(st.session_state.candidate_info)
                            response += "\n\nYour application has been submitted successfully. A recruiter will contact you soon!"
                        except DuplicateCandidateException:
                            response += "\n\nIt looks like you've already completed an assessment with this email address."
                        except Exception as ex:
                            response += f"\n\n❌ An error occurred while saving your application: {str(ex)}"
                        st.session_state.current_stage = "end"
                    message_placeholder.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                
                elif stage == "technical_questions":
                    response, evaluation = result
                    if st.session_state.bot.current_question_idx > 0 and st.session_state.bot.technical_questions:
                        question_idx = st.session_state.bot.current_question_idx - 1
                        question = st.session_state.bot.technical_questions[question_idx]["question"]
                        st.session_state.candidate_info["technical_answers"][question] = evaluation
                        
                        # Immediate feedback if answer score is below threshold (score < 5)
                        if evaluation.get("score", 5) < 5:
                            response += f"\n\nImmediate Feedback: {evaluation.get('feedback', 'Your answer could be improved.')}"
                        
                        next_question_found = False
                        max_questions = min(10, len(st.session_state.bot.technical_questions))
                        for i in range(st.session_state.bot.current_question_idx, len(st.session_state.bot.technical_questions)):
                            next_q = st.session_state.bot.technical_questions[i]["question"]
                            if next_q not in st.session_state.bot.asked_questions and len(st.session_state.bot.asked_questions) < max_questions:
                                response += f"\n\nNext question:\n\nQuestion {len(st.session_state.bot.asked_questions)+1}: {next_q}"
                                st.session_state.bot.current_question_idx = i + 1
                                st.session_state.bot.asked_questions.add(next_q)
                                next_question_found = True
                                break
                        
                        if not next_question_found:
                            response += "\n\nThank you for answering all technical questions. We are now concluding your interview."
                            try:
                                bot.save_candidate_data(st.session_state.candidate_info)
                                response += "\n\nYour application has been submitted successfully. A recruiter will contact you soon!"
                            except DuplicateCandidateException:
                                response += "\n\nIt looks like you've already completed an assessment with this email address."
                            except Exception as ex:
                                response += f"\n\n❌ An error occurred while saving your application: {str(ex)}"
                            st.session_state.current_stage = "end"
                    else:
                        response += "\n\nThere was an error processing your answer."
                    message_placeholder.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                
                else:
                    response = bot.generate_bot_response(prompt, stage, st.session_state.candidate_info)
                    message_placeholder.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    next_stage_index = TalentScoutBot.STAGES.index(stage) + 1
                    if next_stage_index < len(TalentScoutBot.STAGES):
                        st.session_state.current_stage = TalentScoutBot.STAGES[next_stage_index]
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                
if __name__ == "__main__":
    main()
