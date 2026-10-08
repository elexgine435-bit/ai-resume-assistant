import os

import streamlit as st
from google import genai

from ats import analyze, extract_text

st.set_page_config(page_title="ATS Resume Checker", page_icon="📄", layout="centered")

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.getenv("GEMINI_API_KEY")


@st.cache_resource
def get_client(key: str):
    return genai.Client(api_key=key)


def color(score: int) -> str:
    return "🟢" if score >= 75 else "🟡" if score >= 50 else "🔴"


st.title("📄 ATS Resume Checker")
st.caption("Upload your resume to get an ATS score and concrete improvements.")

key = get_api_key()
if not key:
    st.error("GEMINI_API_KEY is not set. Add it to .streamlit/secrets.toml or Streamlit Cloud secrets.")
    st.stop()

file = st.file_uploader("Resume (PDF, DOCX or TXT)", type=["pdf", "docx", "txt"])
jd = st.text_area("Job description (optional, improves keyword matching)", height=150)

if st.button("Analyze", type="primary", disabled=file is None):
    try:
        with st.spinner("Reading resume..."):
            text = extract_text(file.name, file.getvalue())
        with st.spinner("Analyzing with Gemini..."):
            st.session_state["result"] = analyze(get_client(key), MODEL, text, jd)
    except ValueError as e:
        st.warning(str(e))
    except Exception as e:
        st.error(f"Analysis failed: {e}")

r = st.session_state.get("result")
if r:
    st.divider()
    st.metric("ATS Score", f"{r['overall_score']} / 100")
    st.progress(r["overall_score"] / 100)
    st.write(r["summary"])

    if r["section_scores"]:
        st.subheader("Score breakdown")
        for k, v in r["section_scores"].items():
            st.write(f"{color(v)} **{k.replace('_', ' ').title()}** — {v}")
            st.progress(v / 100)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("✅ Strengths")
        for s in r["strengths"]:
            st.markdown(f"- {s}")
    with c2:
        st.subheader("⚠️ Weaknesses")
        for w in r["weaknesses"]:
            st.markdown(f"- {w}")

    if r["missing_keywords"]:
        st.subheader("🔑 Missing keywords")
        st.write(", ".join(f"`{k}`" for k in r["missing_keywords"]))

    st.subheader("🛠 Improvements")
    for i in r["improvements"]:
        with st.expander(f"{i.get('section', 'General')}: {i.get('issue', '')[:80]}"):
            st.markdown(f"**Fix:** {i.get('fix', '')}")
            if i.get("example"):
                st.markdown(f"**Example:** _{i['example']}_")
