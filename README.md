import io
import json
import re

from pypdf import PdfReader
from docx import Document

MAX_CHARS = 20000

PROMPT = """You are an expert ATS (Applicant Tracking System) analyst and resume coach.
Evaluate the resume below{jd_part}. Be strict, specific and honest.

Return ONLY valid JSON with exactly this schema:
{{
  "overall_score": <integer 0-100>,
  "section_scores": {{
    "formatting": <0-100>, "keywords": <0-100>, "experience_impact": <0-100>,
    "skills": <0-100>, "education": <0-100>, "readability": <0-100>
  }},
  "summary": "<2-3 sentence verdict>",
  "strengths": ["..."],
  "weaknesses": ["..."],
  "missing_keywords": ["..."],
  "improvements": [
    {{"section": "<section name>", "issue": "<problem>", "fix": "<concrete fix>",
      "example": "<rewritten example line>"}}
  ]
}}

Scoring: penalise missing quantified results, weak verbs, missing sections
(contact, summary, experience, skills, education), tables/columns/graphics that
break parsing, typos, and (if a job description is given) missing keywords.

{jd_block}RESUME:
\"\"\"
{resume}
\"\"\"
"""


def extract_text(filename: str, data: bytes) -> str:
    try:
        return _extract(filename, data)
    except ValueError:
        raise
    except Exception:
        raise ValueError("Couldn't open this file. It may be corrupted or password-protected.")


def _extract(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((p.extract_text() or "") for p in reader.pages)
    elif name.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        parts = [p.text for p in doc.paragraphs]
        for t in doc.tables:
            for row in t.rows:
                parts.append(" | ".join(c.text for c in row.cells))
        text = "\n".join(parts)
    elif name.endswith(".txt"):
        text = data.decode("utf-8", errors="ignore")
    else:
        raise ValueError("Unsupported file type. Upload a PDF, DOCX or TXT.")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) < 50:
        raise ValueError(
            "Couldn't read enough text. The file may be a scanned image; "
            "upload a text-based PDF or DOCX."
        )
    return text[:MAX_CHARS]


def build_prompt(resume: str, job_description: str = "") -> str:
    jd = job_description.strip()
    return PROMPT.format(
        jd_part=" against the target job description" if jd else "",
        jd_block=f'JOB DESCRIPTION:\n"""\n{jd[:8000]}\n"""\n\n' if jd else "",
        resume=resume,
    )


def parse_response(raw: str) -> dict:
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            raise ValueError("Model did not return JSON.")
        data = json.loads(m.group(0))

    def clamp(v):
        try:
            return max(0, min(100, int(round(float(v)))))
        except (TypeError, ValueError):
            return 0

    sections = data.get("section_scores") or {}
    return {
        "overall_score": clamp(data.get("overall_score")),
        "section_scores": {k: clamp(v) for k, v in sections.items()},
        "summary": str(data.get("summary", "")),
        "strengths": list(data.get("strengths") or []),
        "weaknesses": list(data.get("weaknesses") or []),
        "missing_keywords": list(data.get("missing_keywords") or []),
        "improvements": [i for i in (data.get("improvements") or []) if isinstance(i, dict)],
    }


def analyze(client, model: str, resume: str, job_description: str = "") -> dict:
    from google.genai import types

    resp = client.models.generate_content(
        model=model,
        contents=build_prompt(resume, job_description),
        config=types.GenerateContentConfig(
            temperature=0.2, response_mime_type="application/json"
        ),
    )
    return parse_response(resp.text)
