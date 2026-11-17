# from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser
import ast
import os
import json
import re
from dotenv import load_dotenv

load_dotenv()

DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"

response_schemas = [
    ResponseSchema(name="overall_score", description="A score from 1-10 rating the resume's overall quality"),
    ResponseSchema(name="strengths", description="A bulleted list of what's good about the resume, as a single string with newline-separated bullets"),
    ResponseSchema(name="weaknesses", description="A bulleted list of what's missing or weak, as a single string with newline-separated bullets"),
    ResponseSchema(name="suggestions", description="A bulleted list of concrete improvement suggestions, as a single string with newline-separated bullets"),
    ResponseSchema(name="ats_keywords", description="A bulleted list of important keywords missing for ATS optimization, as a single string with newline-separated bullets"),
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = parser.get_format_instructions()
REQUIRED_KEYS = [schema.name for schema in response_schemas]

prompt = ChatPromptTemplate.from_template(
    """You are an expert resume reviewer and career coach with 15+ years
of experience in technical recruiting and HR.

Analyze the following resume and provide structured, actionable feedback.
Be specific — reference actual content from the resume rather than generic advice.

Resume:
{resume_text}

{format_instructions}

IMPORTANT: Respond with ONLY the JSON object. Do not include any preamble,
explanation, markdown code fences, or text before or after the JSON.
"""
)

def get_llm(model_name: str | None = None):
    return ChatGroq(
        model=model_name or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )


def _is_quota_error(error: Exception) -> bool:
    message = str(error).lower()
    return "insufficient_quota" in message or "error code: 429" in message


def _is_auth_error(error: Exception) -> bool:
    message = str(error).lower()
    return (
        "invalid_api_key" in message
        or "invalid api key" in message
        or "error code: 401" in message
        or "unauthorized" in message
    )


def _is_model_decommissioned_error(error: Exception) -> bool:
    message = str(error).lower()
    return "model_decommissioned" in message or "decommissioned" in message


def _extract_json(raw_text: str) -> dict:
    """Try multiple strategies to pull valid JSON out of the model's raw output."""
    if not raw_text or not raw_text.strip():
        raise ValueError("Model returned an empty response.")

    text = raw_text.strip()

    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Some models return Python-style dict strings with single quotes.
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, SyntaxError):
        pass

    # Try to find the first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(match.group())
                if isinstance(parsed, dict):
                    return parsed
            except (ValueError, SyntaxError):
                pass

    raise ValueError(f"Could not extract valid JSON. Raw model output was: {raw_text[:500]!r}")


def _normalize_review_result(result: dict) -> dict:
    normalized = {}
    for key in REQUIRED_KEYS:
        value = result.get(key, "")
        if value is None:
            value = ""
        normalized[key] = str(value)

    score = normalized.get("overall_score", "").strip()
    normalized["overall_score"] = score if score else "N/A"
    return normalized


def _invoke_review(resume_text: str, model_name: str) -> dict:
    llm = get_llm(model_name)
    messages = prompt.format_messages(
        resume_text=resume_text,
        format_instructions=format_instructions,
    )
    response = llm.invoke(messages)
    raw_content = response.content

    # Try the structured parser first
    try:
        parsed = parser.parse(raw_content)
        return _normalize_review_result(parsed)
    except Exception:
        # Fall back to manual JSON extraction
        return _normalize_review_result(_extract_json(raw_content))


def review_resume(resume_text: str) -> dict:
    configured_model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    try:
        return _invoke_review(resume_text, configured_model)
    except Exception as error:
        if _is_model_decommissioned_error(error) and configured_model != DEFAULT_GROQ_MODEL:
            try:
                return _invoke_review(resume_text, DEFAULT_GROQ_MODEL)
            except Exception as fallback_error:
                raise RuntimeError(
                    "The configured Groq model is no longer supported, and the fallback model also failed. "
                    "Set GROQ_MODEL to a current model such as openai/gpt-oss-20b."
                ) from fallback_error
        if _is_model_decommissioned_error(error):
            raise RuntimeError(
                "The configured Groq model is no longer supported. Set GROQ_MODEL to a current model such as openai/gpt-oss-20b."
            ) from error
        if _is_quota_error(error):
            raise RuntimeError(
                "API quota exhausted. Check your plan and billing details, then try again."
            ) from error
        if _is_auth_error(error):
            raise RuntimeError(
                "Invalid Groq API key. Replace GROQ_API_KEY in .env with a new key from console.groq.com, "
                "then restart the server."
            ) from error
        raise RuntimeError(f"Failed to get a valid response from the model: {error}") from error