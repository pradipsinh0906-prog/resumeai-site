import io
import re
from datetime import date

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from resume_parser import extract_text_from_pdf
from review_chain import review_resume

app = FastAPI(title="ResumeAI API")

# Only your website may call this API from a browser.
# (To test index.html straight from your computer, temporarily use ["*"].)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://resumeai-check.netlify.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_BYTES = 10 * 1024 * 1024


def _bullets(text: str) -> list[str]:
    """Turn a newline-separated bullet string into a clean list."""
    items = []
    for line in str(text).splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if line:
            items.append(line)
    return items


def _score_100(raw: str) -> int:
    match = re.search(r"\d+(?:\.\d+)?", str(raw))
    if not match:
        return 0
    value = float(match.group())
    if value <= 10:  # review_chain scores 1-10
        value *= 10
    return max(0, min(100, round(value)))


def _headline(score: int) -> tuple[str, str]:
    if score >= 80:
        return "Strong resume.", "Your resume is in good shape. A few tweaks can make it stand out even more."
    if score >= 60:
        return "Good start. Great potential.", "Your experience stands out. Let’s make the impact clearer."
    if score >= 40:
        return "Needs work.", "The basics are there. Focused edits will lift your score quickly."
    return "Let’s rebuild this.", "Follow the suggestions below to make your resume much stronger."


def _suggestion(line: str) -> dict:
    if ": " in line:
        title, detail = line.split(": ", 1)
        if len(title) <= 70:
            return {"title": title.strip(), "detail": detail.strip()}
    return {"title": line, "detail": ""}


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(resume: UploadFile = File(...), job_description: str = Form("")):
    if not (resume.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    data = resume.file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="This file is over 10 MB. Please upload a smaller PDF.")

    try:
        text = extract_text_from_pdf(io.BytesIO(data))
    except Exception:
        raise HTTPException(status_code=422, detail="Could not read this PDF. Try exporting it again.")

    if not text:
        raise HTTPException(
            status_code=422,
            detail="No text found in this PDF. It may be a scanned image. Upload a text-based PDF.",
        )

    # The model's own knowledge is older than today, so tell it the real date.
    # Otherwise it flags recent dates on the resume as "future dates".
    today = date.today().strftime("%B %d, %Y")
    text = f"(Today's date is {today}. Treat any date up to today as past, not future.)\n\n{text}"

    jd = job_description.strip()
    if jd:
        text = f"{text}\n\n--- TARGET JOB DESCRIPTION (also judge how well the resume matches it) ---\n{jd}"

    # The model sometimes returns an empty reply. Retry a few times before giving up.
    result = None
    last_error = ""
    for _ in range(3):
        try:
            result = review_resume(text)
            break
        except RuntimeError as e:
            last_error = str(e)
            if "empty response" not in last_error.lower():
                break
    if result is None:
        raise HTTPException(
            status_code=502,
            detail="The AI did not respond this time. Please click Analyze again."
            if "empty response" in last_error.lower()
            else last_error,
        )

    score = _score_100(result.get("overall_score", ""))
    headline, summary = _headline(score)
    suggestions = [_suggestion(s) for s in _bullets(result.get("suggestions", ""))]
    missing = _bullets(result.get("ats_keywords", ""))
    if missing:
        suggestions.append({"title": "Add missing keywords", "detail": ", ".join(missing[:8])})

    return {
        "score": score,
        "headline": headline,
        "summary": summary,
        "strengths": _bullets(result.get("strengths", ""))[:5],
        "weaknesses": _bullets(result.get("weaknesses", ""))[:5],
        "suggestions": suggestions[:6],
    }
