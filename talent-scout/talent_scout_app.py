import os
import streamlit as st
import logging
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Tuple, Union
from pathlib import Path

# Core imports
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_community.llms import HuggingFaceEndpoint  # Open-source LLM
from email_validator import validate_email, EmailNotValidError
import phonenumbers
import pycountry
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv, find_dotenv

# Database operations
from database import create_connection, create_table, insert_candidate, DuplicateCandidateException

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(find_dotenv())

# Check required environment variables
required_env_vars = ["DB_NAME", "DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT", "HF_API_TOKEN"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")

# Helper Functions
def validate_input(input_type: str, value: str) -> Tuple[bool, Any, str]:
    """Unified validation function for different input types"""
    if input_type == "email":
        try:
            validation = validate_email(value, check_deliverability=False)
            return True, validation.normalized, ""
        except EmailNotValidError as e:
            return False, None, str(e)
    
    elif input_type == "phone":
        if not value.strip().startswith("+"):
            return False, None, "Please input your phone number in international format (e.g., +91XXXXXXXXXX)"
        try:
            parsed_number = phonenumbers.parse(value, None)
            if phonenumbers.is_possible_number(parsed_number) and phonenumbers.is_valid_number(parsed_number):
                formatted_number = phonenumbers.format_number(parsed_number, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
                return True, formatted_number, ""
            else:
                return False, None, "Invalid phone number"
        except Exception as e:
            return False, None, str(e)
    
    elif input_type == "experience":
        try:
            clean_exp = ''.join(c for c in value if c.isdigit() or c in ".-")
            exp_value = float(clean_exp)
            if exp_value < 0:
                return False, None, "Experience cannot be negative"
            if exp_value > 50:
                return False, None, "Please enter a realistic experience value (0-50 years)"
            return True, exp_value, ""
        except ValueError:
            return False, None, "Please enter a valid number for years of experience"
    
    elif input_type == "text":
        if len(value.strip()) < 3:
            return False, None, "Please provide a valid input (minimum 3 characters)"
        return True, value.strip(), ""
    
    return False, None, "Invalid input type"

def parse_llm_json(json_text: str, default_value: Any = None) -> Any:
    """Parse JSON response from LLM with error handling"""
    try:
        # Find JSON content if wrapped in markdown code blocks or other text
        if "```json" in json_text:
            start = json_text.find("```json") + 7
            end = json_text.find("```", start)
            json_text = json_text[start:end].strip()
        elif "```" in json_text:
            start = json_text.find("```") + 3
            end = json_text.find("```", start)
            json_text = json_text[start:end].strip()
            
        return json.loads(json_text)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse JSON: {e}")
        return default_value

# TalentScoutBot Class
class TalentScoutBot:
    """Streamlined TalentScout chatbot using open-source LLM instead of OpenAI"""
    
    STAGES = [
        "welcome", "email", "phone", "experience", "position", 
        "location", "tech_stack", "technical_questions", "end"
    ]
    
    def __init__(self):
        self.hf_api_token = os.getenv("HF_API_TOKEN")
        if not self.hf_api_token:
            logger.error("HuggingFace API Token not found in environment variables.")
            raise ValueError("❌ HuggingFace API Token not found in environment variables.")
        
        try:
            # Use HuggingFace endpoint instead of OpenAI
            self.llm = HuggingFaceEndpoint(
            endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
            huggingfacehub_api_token=self.hf_api_token,
            task="text-generation"  # Add this line
            )


            
            self.memory = ConversationBufferMemory(return_messages=True)
            self.conversation = ConversationChain(llm=self.llm, memory=self.memory, verbose=False)
        except Exception as e:
            logger.error(f"Error initializing HuggingFace client: {e}")
            raise ValueError(f"❌ Could not initialize HuggingFace client: {e}")
        
        # Initialize database and other components
        self.db_conn = create_connection()
        if self.db_conn is None:
            logger.error("Could not connect to the database.")
            raise ConnectionError("❌ Could not connect to the database.")
        
        try:
            create_table(self.db_conn)
        except Exception as e:
            logger.error(f"Error creating table: {e}")
            raise e
            
        self.technical_questions = []
        self.current_question_idx = 0
        self.asked_questions = set()
        self.tech_question_cache = {}
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_response(self, user_input: str, stage: str, candidate_info: Dict[str, Any]) -> Union[str, Tuple[str, Any]]:
        """Generate response based on the current stage"""
        try:
            prompt = self._create_stage_prompt(stage, candidate_info, user_input)
            response = self.conversation.predict(input=prompt)
            
            # Handle special stages with additional processing
            if stage == "tech_stack":
                tech_stack = self._extract_tech_stack(user_input)
                return response, tech_stack
            elif stage == "technical_questions":
                if self.current_question_idx > 0 and self.technical_questions:
                    evaluation = self._evaluate_technical_answer(
                        question=self.technical_questions[self.current_question_idx-1]["question"],
                        answer=user_input,
                        tech_stack=candidate_info.get("tech_stack", [])
                    )
                    return response, evaluation
                else:
                    return response, {"score": 5, "feedback": "Could not evaluate answer properly."}
            
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I'm having trouble processing your response. Let's try again. (Error: {str(e)})"
    
    def _create_stage_prompt(self, stage: str, candidate_info: Dict[str, Any], user_input: str) -> str:
        """Create stage-specific prompt for the LLM"""
        base_prompt = (
            "You are TalentScout, a professional AI hiring assistant. "
            "Respond in a direct, professional manner without using phrases like 'Dear candidate', 'Sure', or 'Absolutely'. "
            f"The candidate has provided: {user_input}. "
        )
        
        stage_prompts = {
            "welcome": "Thank the candidate for providing their name and ask for their email address.",
            "email": "The candidate provided their email. Ask for their contact phone number.",
            "phone": "The candidate provided their phone number. Validate it and then ask for their years of experience.",
            "experience": "The candidate provided their experience. Ask what position they're applying for.",
            "position": "The candidate provided their position. Ask for their current location.",
            "location": "The candidate provided their location. Ask about their technical skills and tech stack.",
            "tech_stack": f"The candidate mentioned these technologies: {user_input}. Extract the tech stack, thank them professionally, and prepare for technical questions.",
            "technical_questions": "The candidate has answered a technical question. Provide professional feedback and move to the next question.",
            "end": "Thank the candidate for their time and inform them that their application has been received."
        }
        
        return base_prompt + stage_prompts.get(stage, "Continue the conversation in a professional tone.")
    
    def _extract_tech_stack(self, user_input: str) -> List[str]:
        """Extract tech stack from user input using the LLM"""
        try:
            extraction_prompt = (
                f"Extract all technologies and programming languages from this text: '{user_input}'\n\n"
                "Format the result as a JSON array of strings with only the names of the technologies."
            )
            
            response = self.llm(extraction_prompt)
            tech_stack = parse_llm_json(response, [])
            
            if not tech_stack or not isinstance(tech_stack, list):
                # Fallback to basic extraction
                words = user_input.split()
                common_tech = ["python", "javascript", "java", "c++", "react", "node", "sql", "aws", 
                              "docker", "kubernetes", "git", "html", "css", "php", "ruby", "go"]
                tech_stack = [word for word in words if word.lower() in common_tech]
            
            return tech_stack
        except Exception as e:
            logger.error(f"Error extracting tech stack: {e}")
            # Simple fallback
            return ["general programming"]
    
    def _evaluate_technical_answer(self, question: str, answer: str, tech_stack: List[str]) -> Dict[str, Any]:
        """Evaluate technical answer using the LLM"""
        try:
            evaluation_prompt = (
                f"Evaluate this technical answer for a {', '.join(tech_stack)} position:\n\n"
                f"Question: {question}\n"
                f"Answer: {answer}\n\n"
                "Provide a JSON response with the following fields:\n"
                "- score: A number from 1-10\n"
                "- feedback: Constructive feedback about the answer\n"
                "- strengths: What was good about the answer\n"
                "- areas_to_improve: What could be improved\n"
                "- accurate: Boolean indicating if the answer is technically accurate"
            )
            
            response = self.llm(evaluation_prompt)
            evaluation = parse_llm_json(response, {
                "score": 5,
                "feedback": "We had trouble evaluating your answer automatically. A human reviewer will assess it.",
                "strengths": "N/A",
                "areas_to_improve": "N/A",
                "accurate": None
            })
            
            return evaluation
        except Exception as e:
            logger.error(f"Error evaluating technical answer: {e}")
            return {
                "score": 5,
                "feedback": "Evaluation error. A human reviewer will assess your answer.",
                "strengths": "N/A",
                "areas_to_improve": "N/A",
                "accurate": None
            }
    
    def generate_technical_questions(self, tech_stack: List[str], experience_level: float) -> List[Dict[str, Any]]:
        """Generate technical questions based on tech stack and experience level"""
        if not tech_stack:
            tech_stack = ["general programming"]
            
        # Check cache first
        cache_key = f"{'-'.join(sorted(tech_stack))}-{experience_level}"
        if cache_key in self.tech_question_cache:
            return self.tech_question_cache[cache_key]
        
        # Determine difficulty based on experience
        difficulty = "basic" if experience_level < 2 else "intermediate" if experience_level < 5 else "advanced"
        
        try:
            prompt = (
                f"Generate 3 {difficulty} technical questions for a candidate with {experience_level} years "
                f"of experience in {', '.join(tech_stack)}.\n\n"
                "For each question, provide:\n"
                "1. The question text\n"
                "2. Expected answer points\n"
                "3. Technology category\n\n"
                "Format the output as a JSON array of objects with the following structure:\n"
                "[\n"
                "  {\n"
                f'    "question": "Question text",\n'
                f'    "expected_answer": "Key points that should be in the answer",\n'
                f'    "technology": "Specific technology this tests",\n'
                f'    "difficulty": "{difficulty}"\n'
                "  }\n"
                "]"
            )
            
            response = self.llm(prompt)
            questions = parse_llm_json(response, [])
            
            if not questions or len(questions) < 3:
                # Generate fallback questions
                questions = self._generate_default_questions(tech_stack, difficulty)
                
            self.tech_question_cache[cache_key] = questions
            return questions
        except Exception as e:
            logger.error(f"Error generating technical questions: {e}")
            return self._generate_default_questions(tech_stack, difficulty)

    def _generate_default_questions(self, tech_stack: List[str], difficulty: str) -> List[Dict[str, Any]]:
        """Generate default questions when LLM generation fails"""
        if not tech_stack:
            tech_stack = ["general programming"]
            
        # Ensure we have at least 3 questions
        tech_list = tech_stack[:3] if len(tech_stack) >= 3 else tech_stack + ["general"] * (3 - len(tech_stack))
        
        return [
            {
                "question": f"Explain your experience with {tech}" if tech != "general" else "Describe your technical background",
                "expected_answer": "Candidate should demonstrate knowledge and experience",
                "technology": tech,
                "difficulty": difficulty
            } for tech in tech_list
        ]
    
    def calculate_candidate_score(self, candidate_info: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall candidate score based on technical answers and experience"""
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
            
            # Generate feedback
            scoring_prompt = (
                "Generate a brief feedback summary for a candidate with the following profile:\n"
                f"- Position: {candidate_info.get('position', 'Software Developer')}\n"
                f"- Experience: {candidate_info.get('experience', 0)} years\n"
                f"- Tech Stack: {', '.join(candidate_info.get('tech_stack', []))}\n"
                f"- Technical Interview Score: {avg_score}/10\n\n"
                "Provide a brief assessment of their strengths and areas for improvement."
            )
            
            try:
                feedback = self.llm(scoring_prompt)
            except Exception as e:
                logger.error(f"Error generating feedback: {e}")
                feedback = "Automatic feedback generation failed. A human recruiter will review the application."
                
            recommendation = "Strong Match" if overall_score >= 8 else "Potential Match" if overall_score >= 6 else "Not Recommended"
                
            return {
                "technical_score": round(avg_score, 1),
                "experience_score": round(exp_score, 1),
                "overall_score": round(overall_score, 1),
                "feedback": feedback,
                "recommendation": recommendation
            }
        except Exception as e:
            logger.error(f"Error calculating candidate score: {e}")
            return {
                "technical_score": 5.0,
                "experience_score": 5.0,
                "overall_score": 5.0,
                "feedback": "Automatic feedback generation failed. A human recruiter will review the application.",
                "recommendation": "Review Required"
            }
    
    def save_candidate_data(self, candidate_info: Dict[str, Any]) -> bool:
        """Save candidate data to database"""
        required_fields = ["name", "email", "phone", "experience", "position", "location"]
        if not all(key in candidate_info for key in required_fields):
            logger.error("Missing required candidate information")
            raise ValueError("Missing required candidate information")
            
        try:
            # Ensure tech_stack exists
            if "tech_stack" not in candidate_info or not candidate_info["tech_stack"]:
                candidate_info["tech_stack"] = ["Not specified"]
                
            # Generate simple embeddings (placeholder)
            candidate_info["embeddings"] = [0.0] * 10
                
            # Calculate evaluation
            candidate_info["evaluation"] = self.calculate_candidate_score(candidate_info)
                
            # Save to database
            insert_candidate(self.db_conn, candidate_info)
            return True
        except DuplicateCandidateException as e:
            logger.warning(f"Duplicate candidate: {e}")
            raise e
        except Exception as e:
            logger.error(f"Error saving candidate data: {e}")
            raise e

# Main Streamlit UI
def main():
    st.set_page_config(
        page_title="TalentScout - AI Hiring Assistant",
        page_icon="👔",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar
    with st.sidebar:
        st.title("TalentScout")
        st.subheader("AI Hiring Assistant")
        st.write("---")
        st.write("This application helps screen candidates through an interactive chat interface.")
        
        if st.button("Reset Conversation"):
            st.session_state.clear()
            st.rerun()

            
        st.write("### System Status")
        try:
            db_conn = create_connection()
            if db_conn:
                st.success("✅ System Online")
            else:
                st.error("❌ Database Connection Failed")
        except Exception as e:
            st.error(f"❌ Database Error: {str(e)}")
            
        # Show progress if interview is in progress
        if "current_stage" in st.session_state and "bot" in st.session_state:
            current_stage = st.session_state.current_stage
            progress = (TalentScoutBot.STAGES.index(current_stage) + 1) / len(TalentScoutBot.STAGES)
            st.write("### Interview Progress")
            st.progress(progress)
    
    # Main content
    st.title("TalentScout: AI Hiring Assistant")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    if "bot" not in st.session_state:
        # Check environment
        if not os.getenv("HF_API_TOKEN"):
            st.warning("⚠️ HuggingFace API Token not found. Please set it in your environment variables.")
            st.stop()
            
        try:
            # Initialize bot
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
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # End interview if at end stage
    if st.session_state.get("current_stage", "") == "end":
        st.markdown("## Interview Concluded")
        st.write("Thank you for completing your interview. Your application has been submitted.")
        st.stop()
    
    
    # Handle user input
    prompt = st.chat_input("Your response...")
    if prompt is not None:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
            
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("⏳ Processing...")
            
            try:
                stage = st.session_state.current_stage
                bot = st.session_state.bot
                
                # Handle stage-specific validation
                valid_input = True
                error_message = ""
                
                if stage == "welcome":
                    is_valid, name, error = validate_input("text", prompt)
                    if not is_valid:
                        valid_input = False
                        error_message = error
                    else:
                        st.session_state.candidate_info["name"] = name
                        response = f"Thank you, {name}. Please provide your email address."
                        message_placeholder.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.session_state.current_stage = "email"
                        valid_input = False  # Skip general processing
                        return
                        
                elif stage == "email":
                    is_valid, email, error = validate_input("email", prompt)
                    if not is_valid:
                        valid_input = False
                        error_message = error
                    else:
                        st.session_state.candidate_info["email"] = email
                        response = "Could you please provide your contact phone number? (Format example: +91XXXXXXXXXX)"
                        message_placeholder.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.session_state.current_stage = "phone"
                        valid_input = False  # Skip general processing
                        return
                        
                elif stage == "phone":
                    is_valid, phone, error = validate_input("phone", prompt)
                    if not is_valid:
                        valid_input = False
                        error_message = error
                    else:
                        st.session_state.candidate_info["phone"] = phone
                        response = f"Thank you, your phone number {phone} has been recorded. Can you please share how many years of experience you have in your field?"
                        message_placeholder.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.session_state.current_stage = "experience"
                        valid_input = False  # Skip general processing
                        return
                        
                elif stage == "experience":
                    is_valid, exp_value, error = validate_input("experience", prompt)
                    if not is_valid:
                        valid_input = False
                        error_message = error
                    else:
                        st.session_state.candidate_info["experience"] = exp_value
                        
                elif stage == "position":
                    is_valid, position, error = validate_input("text", prompt)
                    if not is_valid:
                        valid_input = False
                        error_message = error
                    else:
                        st.session_state.candidate_info["position"] = position
                        
                elif stage == "location":
                    is_valid, location, error = validate_input("text", prompt)
                    if not is_valid:
                        valid_input = False
                        error_message = error
                    else:
                        st.session_state.candidate_info["location"] = location
                
                # Handle validation errors
                if not valid_input:
                    message_placeholder.markdown(f"❌ {error_message}")
                    return
                
                # Process response based on stage
                result = bot.generate_response(prompt, stage, st.session_state.candidate_info)
                
                if stage == "tech_stack":
                    response, tech_stack = result
                    st.session_state.candidate_info["tech_stack"] = tech_stack
                    
                    # Generate technical questions
                    experience = st.session_state.candidate_info.get("experience", 0)
                    st.session_state.bot.technical_questions = bot.generate_technical_questions(tech_stack, experience)
                    st.session_state.bot.current_question_idx = 0
                    st.session_state.bot.asked_questions = set()
                    
                    # Start technical questions if available
                    if st.session_state.bot.technical_questions:
                        first_q = st.session_state.bot.technical_questions[0]["question"]
                        response += f"\n\nLet's start with some technical questions:\n\nQuestion 1: {first_q}"
                        st.session_state.bot.current_question_idx = 1
                        st.session_state.bot.asked_questions.add(first_q)
                        st.session_state.current_stage = "technical_questions"
                    else:
                        response += "\n\nI apologize, but I'm having trouble generating technical questions. Let's proceed with concluding the interview."
                        try:
                            bot.save_candidate_data(st.session_state.candidate_info)
                            response += "\n\nYour application has been submitted successfully. A recruiter will contact you soon."
                        except Exception as ex:
                            response += f"\n\n❌ An error occurred: {str(ex)}"
                        st.session_state.current_stage = "end"
                        st.rerun()

                        
                    message_placeholder.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                elif stage == "technical_questions":
                    response, evaluation = result
                    if st.session_state.bot.current_question_idx > 0 and st.session_state.bot.technical_questions:
                        # Save evaluation
                        question_idx = st.session_state.bot.current_question_idx - 1
                        question = st.session_state.bot.technical_questions[question_idx]["question"]
                        user_answer = prompt  # the user's typed answer
                        # Wrap the raw answer with the evaluation
                        st.session_state.candidate_info["technical_answers"][question] = {
                        "user_answer": user_answer,
                        **evaluation
}
                        
                        # Provide feedback for low scores
                        if evaluation.get("score", 5) < 5:
                            response += f"\n\nFeedback: {evaluation.get('feedback', 'Your answer could be improved.')}"
                        
                        # Check if we've asked all questions
                        max_questions = min(3, len(st.session_state.bot.technical_questions))
                        if len(st.session_state.bot.asked_questions) >= max_questions:
                            response += "\n\nThank you for answering all technical questions. We are now concluding your interview."
                            try:
                                bot.save_candidate_data(st.session_state.candidate_info)
                                response += "\n\nYour application has been submitted successfully. A recruiter will contact you soon."
                            except Exception as ex:
                                response += f"\n\n❌ An error occurred: {str(ex)}"
                                
                            st.session_state.messages.append({"role": "assistant", "content": response})
    
                            st.session_state.current_stage = "end"
                            st.rerun()

                        else:
                            # Ask next question
                            for i in range(st.session_state.bot.current_question_idx, len(st.session_state.bot.technical_questions)):
                                next_q = st.session_state.bot.technical_questions[i]["question"]
                                if next_q not in st.session_state.bot.asked_questions:
                                    response += f"\n\nNext question:\n\nQuestion {len(st.session_state.bot.asked_questions)+1}: {next_q}"
                                    st.session_state.bot.current_question_idx = i + 1
                                    st.session_state.bot.asked_questions.add(next_q)
                                    break
                    
                    message_placeholder.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                
                else:
                    # General processing for other stages
                    response = bot.generate_response(prompt, stage, st.session_state.candidate_info)
                    message_placeholder.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Move to next stage
                    next_stage_index = TalentScoutBot.STAGES.index(stage) + 1
                    if next_stage_index < len(TalentScoutBot.STAGES):
                        st.session_state.current_stage = TalentScoutBot.STAGES[next_stage_index]
                
            except Exception as e:
                message_placeholder.markdown(f"❌ An error occurred: {str(e)}")
                logger.error(f"Error in main UI loop: {e}")

if __name__ == "__main__":
    main()