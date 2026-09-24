"""
MemeGen X — Enterprise AI Meme Studio
Single-file full-stack Streamlit application.

Install:
    pip install streamlit groq pillow opencv-python numpy python-dotenv requests

Optional:
    pip install duckduckgo-search

Run:
    streamlit run app.py

Streamlit Cloud Secrets:
    SUPABASE_URL = "https://almmvgiimkftvgdsiiko.supabase.co"
    SUPABASE_PUBLISHABLE_KEY = "sb_publishable_..."
    GROQ_API_KEY = "gsk_..."

Environment variables are also supported locally.

Current Groq model choices used here:
    MAIN_MODEL  = openai/gpt-oss-120b
    FAST_MODEL  = openai/gpt-oss-20b
    VISION_MODEL = qwen/qwen3.8-27b
    SAFETY_MODEL = openai/gpt-oss-safeguard-20b
    STT_MODEL   = whisper-large-v3-turbo

The application is intentionally model-configurable so model IDs can be
changed without rewriting the application.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import sqlite3
import time
import uuid
import urllib.parse
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

try:
    import cv2
except Exception:
    cv2 = None

try:
    import requests
except Exception:
    requests = None

try:
    from groq import Groq
except Exception:
    Groq = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    from supabase import create_client
except Exception:
    create_client = None


# ============================================================================
# CONFIG
# ============================================================================

st.set_page_config(
    page_title="MemeGen X — AI Creative Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "MemeGen X"
DB_PATH = Path("memegen_x.db")

def secret(name: str, default: str = "") -> str:
    """Read Streamlit Cloud Secrets first, then environment variables."""
    try:
        value = st.secrets.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    except Exception:
        pass
    return os.getenv(name, default)


SUPABASE_URL = secret(
    "SUPABASE_URL",
    "https://almmvgiimkftvgdsiiko.supabase.co",
)
# Preferred current Supabase key. Also accepts legacy anon key for compatibility.
SUPABASE_PUBLISHABLE_KEY = secret(
    "SUPABASE_PUBLISHABLE_KEY",
    secret("SUPABASE_ANON_KEY", ""),
)
SUPABASE_SECRET_KEY = secret("SUPABASE_SECRET_KEY", "")

ASSET_DIR = Path("memegen_assets")
ASSET_DIR.mkdir(exist_ok=True)

MAIN_MODEL = secret("MEMEGEN_MAIN_MODEL", "openai/gpt-oss-120b")
FAST_MODEL = secret("MEMEGEN_FAST_MODEL", "openai/gpt-oss-20b")
VISION_MODEL = secret("MEMEGEN_VISION_MODEL", "qwen/qwen3.8-27b")
SAFETY_MODEL = secret("MEMEGEN_SAFETY_MODEL", "openai/gpt-oss-safeguard-20b")
STT_MODEL = secret("MEMEGEN_STT_MODEL", "whisper-large-v3-turbo")

GROQ_API_KEY = secret("GROQ_API_KEY", "")

if Groq and GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    groq_client = None

@st.cache_resource
def get_supabase_client():
    if create_client is None:
        return None

    # Browser/client-safe key. Never use the sb_secret key in this user path.
    key = SUPABASE_PUBLISHABLE_KEY
    if not SUPABASE_URL or not key:
        return None

    try:
        return create_client(SUPABASE_URL, key)
    except Exception:
        return None

supabase_client = get_supabase_client()

SUPABASE_READY = bool(
    create_client is not None
    and SUPABASE_URL
    and SUPABASE_PUBLISHABLE_KEY
    and supabase_client is not None
)


# ============================================================================
# DATABASE
# ============================================================================

def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    conn = db()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT,
            role TEXT NOT NULL DEFAULT 'creator',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS memes (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            project_id TEXT,
            prompt TEXT NOT NULL,
            caption TEXT,
            language TEXT,
            tone TEXT,
            template TEXT,
            image_path TEXT,
            model TEXT,
            latency_ms INTEGER,
            quality_score REAL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS evaluations (
            id TEXT PRIMARY KEY,
            meme_id TEXT NOT NULL,
            relevance REAL,
            humor REAL,
            language REAL,
            originality REAL,
            visual REAL,
            safety REAL,
            overall REAL,
            evaluator_model TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(meme_id) REFERENCES memes(id)
        );

        CREATE TABLE IF NOT EXISTS audit_logs (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            event TEXT NOT NULL,
            metadata TEXT,
            created_at TEXT NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()


init_db()


# ============================================================================
# SESSION
# ============================================================================

defaults = {
    "authenticated": False,
    "user_id": None,
    "email": None,
    "role": "creator",
    "page": "Create",
    "prompt": "",
    "language": "Tanglish",
    "tone": "Savage",
    "creativity": 0.78,
    "last_result": None,
    "history_refresh": 0,
    "project": "Default Workspace",
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================================
# UI
# ============================================================================

def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

        :root {
            --bg: #07080d;
            --panel: rgba(17, 20, 30, .76);
            --panel2: rgba(23, 27, 40, .86);
            --line: rgba(255,255,255,.09);
            --text: #f6f7fb;
            --muted: #98a1b3;
            --cyan: #38e8ff;
            --purple: #8b5cf6;
            --green: #47e3a5;
        }

        .stApp {
            background:
              radial-gradient(circle at 5% 0%, rgba(56,232,255,.10), transparent 28%),
              radial-gradient(circle at 95% 5%, rgba(139,92,246,.12), transparent 30%),
              linear-gradient(180deg, #07080d 0%, #0a0d14 100%);
            color: var(--text);
            font-family: Inter, sans-serif;
        }

        header[data-testid="stHeader"] { background: transparent; }
        footer { visibility: hidden; }

        .block-container {
            max-width: 1500px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        h1,h2,h3,h4 {
            font-family: "Space Grotesk", sans-serif !important;
        }

        .brand {
            font-family: "Space Grotesk", sans-serif;
            font-size: 28px;
            font-weight: 800;
            letter-spacing: -.04em;
        }

        .gradient {
            background: linear-gradient(90deg, #38e8ff, #8b5cf6, #ff4fd8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero {
            padding: 26px 30px;
            border: 1px solid var(--line);
            border-radius: 24px;
            background:
                radial-gradient(circle at 80% 20%, rgba(139,92,246,.20), transparent 35%),
                linear-gradient(135deg, rgba(255,255,255,.045), rgba(255,255,255,.018));
            box-shadow: 0 20px 80px rgba(0,0,0,.28);
            margin-bottom: 20px;
        }

        .hero-title {
            font-family: "Space Grotesk", sans-serif;
            font-size: clamp(34px, 5vw, 62px);
            line-height: .98;
            font-weight: 800;
            letter-spacing: -.055em;
            margin-bottom: 12px;
        }

        .muted { color: var(--muted); }

        .card {
            border: 1px solid var(--line);
            border-radius: 20px;
            background: var(--panel);
            padding: 20px;
            box-shadow: 0 15px 45px rgba(0,0,0,.18);
            backdrop-filter: blur(18px);
        }

        .metric-card {
            border: 1px solid var(--line);
            border-radius: 16px;
            background: rgba(255,255,255,.035);
            padding: 16px;
        }

        .metric-value {
            font-size: 28px;
            font-weight: 800;
            font-family: "Space Grotesk", sans-serif;
        }

        .pill {
            display:inline-block;
            padding: 5px 10px;
            border-radius: 999px;
            background: rgba(56,232,255,.09);
            border: 1px solid rgba(56,232,255,.20);
            color: #aaf6ff;
            font-size: 12px;
            font-weight: 700;
            margin-right: 5px;
        }

        .score-bar {
            height: 8px;
            border-radius: 99px;
            background: rgba(255,255,255,.07);
            overflow: hidden;
            margin: 6px 0 13px;
        }

        .score-fill {
            height:100%;
            border-radius:99px;
            background: linear-gradient(90deg,#38e8ff,#8b5cf6);
        }

        .stButton > button {
            border-radius: 12px !important;
            border: 1px solid rgba(255,255,255,.10) !important;
            background: linear-gradient(135deg, rgba(56,232,255,.13), rgba(139,92,246,.15)) !important;
            color: white !important;
            font-weight: 700 !important;
            transition: .2s ease;
        }

        .stButton > button:hover {
            border-color: rgba(56,232,255,.45) !important;
            transform: translateY(-1px);
            box-shadow: 0 8px 28px rgba(56,232,255,.10);
        }

        textarea, input {
            border-radius: 12px !important;
        }

        section[data-testid="stSidebar"] {
            background: rgba(7,8,13,.82);
            border-right: 1px solid var(--line);
        }

        div[data-testid="stMetric"] {
            background: rgba(255,255,255,.025);
            padding: 12px;
            border: 1px solid var(--line);
            border-radius: 14px;
        }

        .tiny {
            font-size: 11px;
            color: var(--muted);
        }

        .status {
            display:flex;
            align-items:center;
            gap:8px;
            color:#b9c2d0;
            font-size:13px;
        }

        .dot {
            width:8px;
            height:8px;
            border-radius:50%;
            background:#47e3a5;
            box-shadow:0 0 12px #47e3a5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


# ============================================================================
# AUTH — Supabase Auth
# ============================================================================

def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_user(email: str, password: str) -> tuple[bool, str]:
    """Create the user in Supabase Auth.

    Passwords are handled by Supabase Auth. We deliberately do not hash or
    store user passwords in the local SQLite database.
    """
    email = email.strip().lower()

    if not email or "@" not in email:
        return False, "Enter a valid email address."

    if len(password) < 8:
        return False, "Password must contain at least 8 characters."

    if supabase_client is None:
        return False, (
            "Supabase is not configured. Check SUPABASE_URL and "
            "SUPABASE_PUBLISHABLE_KEY in Streamlit Secrets."
        )

    try:
        result = supabase_client.auth.sign_up(
            {"email": email, "password": password}
        )

        if not result.user:
            return False, "Supabase did not create the account."

        uid = str(result.user.id)

        # Local cache only — never store the password.
        conn = db()
        conn.execute(
            """INSERT OR IGNORE INTO users
               (id, email, password_hash, role, created_at)
               VALUES (?, ?, NULL, ?, ?)""",
            (uid, email, "creator", now()),
        )
        conn.commit()
        conn.close()

        if result.session:
            st.session_state.authenticated = True
            st.session_state.user_id = uid
            st.session_state.email = email
            st.session_state.role = "creator"
            return True, "Account created and signed in."

        return True, (
            "Account created. Check your email to confirm the account, "
            "then sign in."
        )

    except Exception as exc:
        return False, f"Supabase signup failed: {exc}"


def login_user(email: str, password: str) -> tuple[bool, str]:
    email = email.strip().lower()

    if not email or not password:
        return False, "Enter your email and password."

    if supabase_client is None:
        return False, (
            "Supabase is not configured. Check SUPABASE_URL and "
            "SUPABASE_PUBLISHABLE_KEY in Streamlit Secrets."
        )

    try:
        result = supabase_client.auth.sign_in_with_password(
            {"email": email, "password": password}
        )

        if not result.user:
            return False, "Invalid email or password."

        uid = str(result.user.id)

        conn = db()
        conn.execute(
            """INSERT OR IGNORE INTO users
               (id, email, password_hash, role, created_at)
               VALUES (?, ?, NULL, ?, ?)""",
            (uid, email, "creator", now()),
        )
        conn.commit()
        conn.close()

        st.session_state.authenticated = True
        st.session_state.user_id = uid
        st.session_state.email = email
        st.session_state.role = "creator"

        return True, "Authenticated with Supabase."

    except Exception as exc:
        return False, f"Supabase login failed: {exc}"


def auth_page() -> None:
    st.markdown(
        """
        <div style="max-width:900px;margin:8vh auto 0;">
          <div class="hero">
            <div class="pill">AI CREATIVE PLATFORM</div>
            <div class="hero-title">Create memes with an <span class="gradient">AI creative engine.</span></div>
            <div class="muted">
              Multimodal generation, candidate ranking, computer vision layout,
              quality evaluation and persistent workspace.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    a, b, c = st.columns([1, 1.3, 1])
    with b:
        tab1, tab2 = st.tabs(["Sign in", "Create account"])

        with tab1:
            with st.form("login"):
                email = st.text_input("Email", placeholder="you@company.com")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button(
                    "Enter workspace", use_container_width=True
                )
                if submit:
                    ok, msg = login_user(email, password)
                    if ok:
                        st.rerun()
                    else:
                        st.error(msg)

        with tab2:
            with st.form("signup"):
                email = st.text_input(
                    "Work email", placeholder="you@company.com"
                )
                password = st.text_input("Password", type="password")
                password2 = st.text_input(
                    "Confirm password", type="password"
                )
                submit = st.form_submit_button(
                    "Create workspace", use_container_width=True
                )
                if submit:
                    if password != password2:
                        st.error("Passwords do not match.")
                    else:
                        ok, msg = create_user(email, password)
                        if ok:
                            st.success(msg)
                        else:
                            st.error(msg)


if not st.session_state.authenticated:
    auth_page()
    st.stop()


# ============================================================================
# HELPERS
# ============================================================================

def log_event(event: str, metadata: dict[str, Any] | None = None) -> None:
    conn = db()
    conn.execute(
        "INSERT INTO audit_logs(id,user_id,event,metadata,created_at) VALUES(?,?,?,?,?)",
        (
            str(uuid.uuid4()),
            st.session_state.user_id,
            event,
            json.dumps(metadata or {}),
            now(),
        ),
    )
    conn.commit()
    conn.close()


def safe_json(text: str, default: dict[str, Any]) -> dict[str, Any]:
    try:
        text = text.strip()
        text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
        text = re.sub(r"```$", "", text).strip()
        return json.loads(text)
    except Exception:
        return default


def b64_image(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def get_font(size: int, bold: bool = True):
    candidates = [
        "Impact.ttf" if bold else "Arial.ttf",
        "impact.ttf" if bold else "arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


# ============================================================================
# AI LAYER
# ============================================================================

@dataclass
class Candidate:
    caption: str
    hook: str = ""
    template_hint: str = "reaction"
    placement: str = "bottom"
    rationale: str = ""


@dataclass
class Scores:
    relevance: float
    humor: float
    language: float
    originality: float
    visual: float
    safety: float
    overall: float


def ai_chat(
    messages: list[dict[str, Any]],
    model: str = MAIN_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 1800,
    reasoning_effort: str = "medium",
    json_mode: bool = False,
) -> str:
    if not groq_client:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to your environment before generating."
        )

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
    }

    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = groq_client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def analyze_prompt(prompt: str, language: str, tone: str) -> dict[str, Any]:
    system = """
You are MemeGen X's intent-analysis engine.
Convert a casual user situation into structured meme-generation intent.
Return ONLY JSON.
"""
    user = f"""
Situation: {prompt}
Language: {language}
Tone: {tone}

Return:
{{
  "topic": "...",
  "emotion": "...",
  "audience": "...",
  "style": "...",
  "keywords": ["..."],
  "visual_concept": "...",
  "safety_risk": "low|medium|high"
}}
"""
    try:
        return safe_json(
            ai_chat(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                model=FAST_MODEL,
                temperature=0.25,
                max_tokens=700,
                reasoning_effort="low",
                json_mode=True,
            ),
            {
                "topic": "general",
                "emotion": "funny",
                "audience": "internet",
                "style": tone,
                "keywords": [],
                "visual_concept": "reaction meme",
                "safety_risk": "low",
            },
        )
    except Exception:
        return {
            "topic": prompt[:80],
            "emotion": "funny",
            "audience": "internet",
            "style": tone,
            "keywords": [],
            "visual_concept": "reaction meme",
            "safety_risk": "low",
        }


def generate_candidates(
    prompt: str,
    language: str,
    tone: str,
    creativity: float,
    intent: dict[str, Any],
) -> list[Candidate]:
    language_rules = {
        "Tanglish": "Use natural Tamil-English internet speech. Do not use formal literary Tamil.",
        "English": "Use natural internet English.",
        "Tamil": "Use natural conversational Tamil.",
        "Hindi": "Use natural conversational Hindi.",
        "German": "Use natural conversational German.",
    }

    system = f"""
You are the primary creative reasoning engine for MemeGen X.
Model role: generate high-quality, original meme candidates.

Rules:
- {language_rules.get(language, language_rules['English'])}
- Tone: {tone}
- Keep captions punchy and meme-native.
- Avoid generic motivational language.
- Avoid protected-person harassment, hateful content, threats or sexual exploitation.
- Generate 6 genuinely different candidates.
- Prefer cultural/contextual specificity.
- Do not copy known meme captions verbatim.
- Output ONLY JSON.
"""

    user = f"""
User situation:
{prompt}

Structured intent:
{json.dumps(intent, ensure_ascii=False)}

Creativity:
{creativity}

Return:
{{
  "candidates": [
    {{
      "caption": "...",
      "hook": "...",
      "template_hint": "reaction|drake|top-bottom|office|college|coding|custom",
      "placement": "top|bottom",
      "rationale": "short reason"
    }}
  ]
}}
"""

    raw = ai_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=MAIN_MODEL,
        temperature=max(.25, min(1.0, creativity)),
        max_tokens=1800,
        reasoning_effort="high",
        json_mode=True,
    )

    data = safe_json(raw, {"candidates": []})
    output: list[Candidate] = []

    for item in data.get("candidates", []):
        caption = str(item.get("caption", "")).strip()
        if caption:
            output.append(
                Candidate(
                    caption=caption[:220],
                    hook=str(item.get("hook", "")),
                    template_hint=str(item.get("template_hint", "reaction")),
                    placement=str(item.get("placement", "bottom")),
                    rationale=str(item.get("rationale", "")),
                )
            )

    return output[:6]


def deterministic_ml_features(
    prompt: str,
    candidate: Candidate,
    language: str,
) -> dict[str, float]:
    """
    Lightweight local ML/ranking feature extractor.
    This is intentionally deterministic and requires no GPU.
    """

    p = prompt.lower()
    c = candidate.caption.lower()

    p_words = set(re.findall(r"\b\w+\b", p))
    c_words = set(re.findall(r"\b\w+\b", c))

    overlap = len(p_words & c_words) / max(1, len(p_words))
    length_quality = 1.0 - min(abs(len(c.split()) - 9) / 14, 1)

    slang_terms = [
        "bro", "vro", "pangu", "maapu", "da", "dei",
        "lol", "bruh", "literally", "fr", "💀", "😂"
    ]
    slang_score = min(sum(1 for x in slang_terms if x in c) / 3, 1)

    punch_score = min(
        (c.count("!") * .10)
        + (c.count("?") * .10)
        + (c.count("💀") * .20)
        + (c.count("😂") * .15)
        + (0.15 if any(w in c for w in ["when", "me", "bro", "me:", "also"]) else 0),
        1,
    )

    repetition_penalty = 0 if len(c_words) >= max(3, len(c.split()) * .55) else .25

    language_bonus = slang_score if language == "Tanglish" else .65

    return {
        "semantic_overlap": min(overlap, 1),
        "length_quality": max(0, length_quality),
        "slang_fit": language_bonus,
        "punch": min(punch_score, 1),
        "repetition": repetition_penalty,
    }


def local_score(candidate: Candidate, prompt: str, language: str) -> Scores:
    f = deterministic_ml_features(prompt, candidate, language)

    relevance = (
        .55 * f["semantic_overlap"]
        + .25 * f["length_quality"]
        + .20 * f["punch"]
    )

    humor = (
        .45 * f["punch"]
        + .30 * f["length_quality"]
        + .25 * f["slang_fit"]
    )

    language_score = .72 + .28 * f["slang_fit"]

    # Originality proxy:
    # penalize repetitive structural patterns without pretending this is
    # a semantic originality detector.
    originality = max(
        .45,
        1.0 - f["repetition"] - (0.10 if candidate.caption.lower().startswith("when when") else 0),
    )

    visual = .55 + .45 * f["length_quality"]
    safety = .98

    overall = (
        .30 * relevance
        + .25 * humor
        + .15 * language_score
        + .15 * originality
        + .10 * visual
        + .05 * safety
    )

    return Scores(
        relevance=round(relevance, 3),
        humor=round(humor, 3),
        language=round(language_score, 3),
        originality=round(originality, 3),
        visual=round(visual, 3),
        safety=round(safety, 3),
        overall=round(overall, 3),
    )


def ai_rerank(
    prompt: str,
    candidates: list[Candidate],
    local_scores: list[Scores],
) -> int:
    """
    Second-stage evaluator.
    Uses GPT-OSS 120B to judge the shortlist after local feature ranking.
    """
    payload = []
    for i, (c, s) in enumerate(zip(candidates, local_scores)):
        payload.append(
            {
                "id": i,
                "caption": c.caption,
                "template": c.template_hint,
                "local_score": asdict(s),
            }
        )

    system = """
You are a strict meme quality evaluator.
Rank candidate captions by contextual relevance, humor, language naturalness,
originality and visual usability.
Do not reward offensive content.
Return ONLY JSON.
"""

    user = f"""
Situation:
{prompt}

Candidates:
{json.dumps(payload, ensure_ascii=False)}

Return:
{{
  "winner": 0,
  "reason": "...",
  "quality": 0.0
}}
"""

    try:
        result = safe_json(
            ai_chat(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                model=MAIN_MODEL,
                temperature=.15,
                max_tokens=500,
                reasoning_effort="high",
                json_mode=True,
            ),
            {"winner": 0, "quality": local_scores[0].overall, "reason": "local rank"},
        )
        idx = int(result.get("winner", 0))
        return max(0, min(idx, len(candidates) - 1))
    except Exception:
        return int(np.argmax([x.overall for x in local_scores]))


def safety_check(text: str) -> bool:
    if not groq_client:
        return True

    try:
        result = ai_chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Classify whether the following meme caption is safe for a "
                        "general consumer application. Return JSON with {\"safe\": true|false}."
                    ),
                },
                {"role": "user", "content": text},
            ],
            model=SAFETY_MODEL,
            temperature=.0,
            max_tokens=120,
            reasoning_effort="low",
            json_mode=True,
        )
        return bool(safe_json(result, {"safe": True}).get("safe", True))
    except Exception:
        return True


def vision_analyze(image_bytes: bytes, instruction: str) -> dict[str, Any]:
    if not groq_client:
        raise RuntimeError("GROQ_API_KEY is missing.")

    encoded = b64_image(image_bytes)

    response = groq_client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the visual intelligence module of an AI meme studio. "
                    "Analyze composition, subjects, emotions, OCR-like text, safe text "
                    "zones and meme potential. Return only JSON."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded}"
                        },
                    },
                ],
            },
        ],
        temperature=.4,
        max_completion_tokens=1200,
        reasoning_effort="medium",
        response_format={"type": "json_object"},
    )

    return safe_json(
        response.choices[0].message.content or "{}",
        {
            "description": "",
            "emotion": "neutral",
            "text_in_image": "",
            "safe_zone": "bottom",
            "meme_ideas": [],
        },
    )


def transcribe_audio(audio_bytes: bytes) -> str:
    if not groq_client:
        raise RuntimeError("GROQ_API_KEY is missing.")

    response = groq_client.audio.transcriptions.create(
        file=("voice.wav", audio_bytes),
        model=STT_MODEL,
        response_format="text",
    )
    return str(response)


# ============================================================================
# COMPUTER VISION / LAYOUT ENGINE
# ============================================================================

def detect_faces(image: Image.Image) -> list[tuple[int, int, int, int]]:
    if cv2 is None:
        return []

    arr = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    if cascade.empty():
        return []

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )
    return [tuple(map(int, f)) for f in faces]


def choose_text_zone(image: Image.Image, preferred: str = "bottom") -> str:
    faces = detect_faces(image)

    if not faces:
        return preferred if preferred in {"top", "bottom"} else "bottom"

    largest = max(faces, key=lambda x: x[2] * x[3])
    _, y, _, h = largest
    center = y + h / 2

    if center < image.height * .43:
        return "bottom"
    if center > image.height * .57:
        return "top"

    return preferred


def wrap_text(draw, text: str, font, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""

    for word in words:
        trial = word if not current else current + " " + word
        box = draw.textbbox((0, 0), trial, font=font)
        if box[2] - box[0] <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines


def burn_meme_text(
    image: Image.Image,
    text: str,
    color: str = "#FFFFFF",
    position: str = "bottom",
) -> Image.Image:
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)

    # Adaptive font-size search.
    size = max(30, int(img.width * .075))
    max_width = int(img.width * .90)

    while size > 18:
        font = get_font(size, True)
        lines = wrap_text(draw, text, font, max_width)
        if len(lines) <= 3:
            break
        size -= 2

    font = get_font(size, True)
    lines = wrap_text(draw, text, font, max_width)

    spacing = max(5, int(size * .12))
    heights = []
    widths = []

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])

    total_h = sum(heights) + spacing * max(0, len(lines) - 1)

    if position == "top":
        y = int(img.height * .045)
    else:
        y = img.height - total_h - int(img.height * .055)

    y = max(8, y)

    outline = max(2, int(size * .055))

    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        x = (img.width - w) // 2

        # Shadow/outline.
        for dx in range(-outline, outline + 1):
            for dy in range(-outline, outline + 1):
                draw.text((x + dx, y + dy), line, font=font, fill="#000000")

        draw.text((x, y), line, font=font, fill=color)
        y += heights[i] + spacing

    return img


def make_gradient_background(
    width: int = 1080,
    height: int = 1080,
    seed: int = 42,
) -> Image.Image:
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 1, height)[:, None]
    x = np.linspace(0, 1, width)[None, :]

    r = (15 + 25 * x + 20 * y).astype(np.uint8)
    g = (17 + 10 * x + 10 * y).astype(np.uint8)
    b = (30 + 50 * (1 - x) + 25 * y).astype(np.uint8)

    arr = np.stack(
        [
            np.broadcast_to(r, (height, width)),
            np.broadcast_to(g, (height, width)),
            np.broadcast_to(b, (height, width)),
        ],
        axis=-1,
    )

    # Subtle noise prevents a flat synthetic background.
    noise = rng.normal(0, 2.0, arr.shape).astype(np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


def render_meme(
    image: Optional[Image.Image],
    caption: str,
    placement: str = "bottom",
    text_color: str = "#FFFFFF",
) -> Image.Image:
    if image is None:
        image = make_gradient_background()

    img = image.convert("RGB")

    # Standardize dimensions for consistent rendering.
    target = 1080
    ratio = target / max(img.width, img.height)
    if ratio < 1:
        img = img.resize((int(img.width * ratio), int(img.height * ratio)))

    # Center-crop square.
    side = min(img.width, img.height)
    left = (img.width - side) // 2
    top = (img.height - side) // 2
    img = img.crop((left, top, left + side, top + side))

    img = ImageEnhance.Contrast(img).enhance(1.04)
    img = ImageEnhance.Sharpness(img).enhance(1.08)

    zone = choose_text_zone(img, placement)
    return burn_meme_text(img, caption, text_color, zone)


# ============================================================================
# PIPELINE
# ============================================================================

def generate_pipeline(
    prompt: str,
    language: str,
    tone: str,
    creativity: float,
    image: Optional[Image.Image] = None,
) -> dict[str, Any]:
    start = time.perf_counter()

    intent = analyze_prompt(prompt, language, tone)

    if image is not None:
        try:
            vision = vision_analyze(
                image_bytes=image_to_bytes(image),
                instruction=(
                    "Analyze this image for meme generation. Identify the subjects, "
                    "emotion, visible text, composition, likely safe caption zones, "
                    "and three meme concepts."
                ),
            )
            intent["vision"] = vision
        except Exception as exc:
            intent["vision_error"] = str(exc)

    candidates = generate_candidates(
        prompt=prompt,
        language=language,
        tone=tone,
        creativity=creativity,
        intent=intent,
    )

    if not candidates:
        raise RuntimeError("The AI returned no usable candidates.")

    local_scores = [
        local_score(c, prompt, language)
        for c in candidates
    ]

    # First-stage ML ranking.
    local_order = np.argsort(
        [-s.overall for s in local_scores]
    ).tolist()

    shortlist_idx = local_order[: min(4, len(local_order))]
    shortlist = [candidates[i] for i in shortlist_idx]
    shortlist_scores = [local_scores[i] for i in shortlist_idx]

    # Second-stage reasoning evaluator.
    winner_shortlist_idx = ai_rerank(prompt, shortlist, shortlist_scores)
    winner_idx = shortlist_idx[winner_shortlist_idx]

    winner = candidates[winner_idx]
    winner_score = local_scores[winner_idx]

    # Safety gate.
    safe = safety_check(winner.caption)

    if not safe:
        safe_indices = [
            i for i, c in enumerate(candidates)
            if safety_check(c.caption)
        ]
        if safe_indices:
            winner_idx = max(
                safe_indices,
                key=lambda i: local_scores[i].overall
            )
            winner = candidates[winner_idx]
            winner_score = local_scores[winner_idx]

    final_image = render_meme(
        image=image,
        caption=winner.caption,
        placement=winner.placement,
    )

    latency = int((time.perf_counter() - start) * 1000)

    return {
        "intent": intent,
        "candidates": candidates,
        "scores": local_scores,
        "winner": winner,
        "winner_score": winner_score,
        "image": final_image,
        "latency_ms": latency,
        "model": MAIN_MODEL,
    }


def image_to_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


# ============================================================================
# SUPABASE CLOUD STORAGE
# ============================================================================

def supabase_upload_image(local_path: Path, object_name: str) -> Optional[str]:
    """Upload to a Supabase Storage bucket named `memes`.
    The bucket should be configured in the Supabase dashboard.
    """
    if supabase_client is None:
        return None

    try:
        with local_path.open("rb") as f:
            supabase_client.storage.from_("memes").upload(
                object_name,
                f.read(),
                {"content-type": "image/png", "upsert": "true"},
            )
        try:
            return supabase_client.storage.from_("memes").get_public_url(object_name)
        except Exception:
            return None
    except Exception as exc:
        log_event("storage.upload_failed", {"error": str(exc)})
        return None


# ============================================================================
# PERSISTENCE
# ============================================================================

def save_meme(result: dict[str, Any], prompt: str, language: str, tone: str) -> str:
    meme_id = str(uuid.uuid4())
    image_path = ASSET_DIR / f"{meme_id}.png"
    result["image"].save(image_path, format="PNG", optimize=True)

    cloud_url = supabase_upload_image(
        image_path,
        f"{st.session_state.user_id}/{meme_id}.png",
    )

    conn = db()
    conn.execute(
        """
        INSERT INTO memes(
            id,user_id,project_id,prompt,caption,language,tone,template,
            image_path,model,latency_ms,quality_score,created_at
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            meme_id,
            st.session_state.user_id,
            None,
            prompt,
            result["winner"].caption,
            language,
            tone,
            result["winner"].template_hint,
            str(image_path),
            result["model"],
            result["latency_ms"],
            result["winner_score"].overall,
            now(),
        ),
    )

    s = result["winner_score"]
    conn.execute(
        """
        INSERT INTO evaluations(
            id,meme_id,relevance,humor,language,originality,
            visual,safety,overall,evaluator_model,created_at
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            str(uuid.uuid4()),
            meme_id,
            s.relevance,
            s.humor,
            s.language,
            s.originality,
            s.visual,
            s.safety,
            s.overall,
            MAIN_MODEL,
            now(),
        ),
    )

    conn.commit()
    conn.close()

    log_event(
        "meme.generated",
        {
            "meme_id": meme_id,
            "model": MAIN_MODEL,
            "latency_ms": result["latency_ms"],
            "quality": s.overall,
            "cloud_storage": bool(cloud_url),
        },
    )

    return meme_id


def history() -> list[sqlite3.Row]:
    conn = db()
    rows = conn.execute(
        """
        SELECT m.*, e.relevance, e.humor, e.language AS language_score,
               e.originality, e.visual, e.safety, e.overall
        FROM memes m
        LEFT JOIN evaluations e ON e.meme_id=m.id
        WHERE m.user_id=?
        ORDER BY m.created_at DESC
        LIMIT 100
        """,
        (st.session_state.user_id,),
    ).fetchall()
    conn.close()
    return rows


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown('<div class="brand">⚡ <span class="gradient">MemeGen X</span></div>', unsafe_allow_html=True)
    st.caption("Enterprise AI Creative Studio")

    st.markdown(
        f"""
        <div class="status">
          <span class="dot"></span>
          {st.session_state.email}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    pages = ["Create", "Vision Roast", "Vault", "Analytics", "AI Lab", "Settings"]
    for p in pages:
        if st.button(
            p,
            use_container_width=True,
            type="primary" if st.session_state.page == p else "secondary",
        ):
            st.session_state.page = p
            st.rerun()

    st.markdown("---")
    st.markdown("### Cloud")
    if supabase_client is not None:
        st.markdown('<div class="status"><span class="dot"></span> Supabase connected</div>', unsafe_allow_html=True)
    else:
        st.caption("Supabase key not configured.")

    st.markdown("### AI Stack")
    st.caption(f"Reasoning: `{MAIN_MODEL}`")
    st.caption(f"Fast: `{FAST_MODEL}`")
    st.caption(f"Vision: `{VISION_MODEL}`")
    st.caption(f"Safety: `{SAFETY_MODEL}`")
    st.caption(f"STT: `{STT_MODEL}`")

    st.markdown("---")
    if st.button("Log out", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.user_id = None
        st.session_state.email = None
        st.rerun()


# ============================================================================
# TOP HEADER
# ============================================================================

st.markdown(
    f"""
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:18px;">
      <div>
        <div class="tiny">WORKSPACE / {st.session_state.page.upper()}</div>
        <div class="brand">AI Creative Studio</div>
      </div>
      <div class="pill">● AI ONLINE</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# CREATE
# ============================================================================

if st.session_state.page == "Create":
    st.markdown(
        """
        <div class="hero">
          <div class="pill">MULTIMODAL MEME ENGINE</div>
          <div class="hero-title">Turn a situation into a <span class="gradient">high-signal meme.</span></div>
          <div class="muted">
            GPT-OSS reasoning → candidate generation → ML ranking → AI evaluation →
            computer-vision layout → final render.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, center, right = st.columns([1.05, 1.55, .95], gap="large")

    with left:
        st.markdown("### Creative brief")

        prompt = st.text_area(
            "Situation",
            value=st.session_state.prompt,
            height=160,
            placeholder=(
                "Example: When your code works perfectly on your laptop "
                "but crashes during the demo..."
            ),
            label_visibility="collapsed",
        )
        st.session_state.prompt = prompt

        c1, c2 = st.columns(2)
        with c1:
            language = st.selectbox(
                "Language",
                ["Tanglish", "English", "Tamil", "Hindi", "German"],
                index=["Tanglish", "English", "Tamil", "Hindi", "German"].index(
                    st.session_state.language
                ),
            )
        with c2:
            tone = st.selectbox(
                "Tone",
                ["Savage", "Wholesome", "Sarcastic", "Chaotic", "Corporate", "Dry"],
                index=["Savage", "Wholesome", "Sarcastic", "Chaotic", "Corporate", "Dry"].index(
                    st.session_state.tone
                ),
            )

        st.session_state.language = language
        st.session_state.tone = tone

        creativity = st.slider(
            "Creativity",
            0.1,
            1.0,
            float(st.session_state.creativity),
            .01,
        )
        st.session_state.creativity = creativity

        uploaded = st.file_uploader(
            "Optional image",
            type=["png", "jpg", "jpeg", "webp"],
        )

        source_image = None
        if uploaded:
            source_image = Image.open(uploaded).convert("RGB")
            st.image(source_image, caption="Reference image", use_container_width=True)

        voice = st.audio_input("Or speak your situation")

        if voice and st.button("Transcribe voice", use_container_width=True):
            with st.spinner("Transcribing..."):
                try:
                    text = transcribe_audio(voice.getvalue())
                    st.session_state.prompt = text
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

        st.markdown("---")

        generate = st.button(
            "⚡ Generate enterprise-quality meme",
            use_container_width=True,
            type="primary",
        )

    with center:
        st.markdown("### Live canvas")

        if generate:
            if not prompt.strip():
                st.warning("Describe the situation first.")
            elif not groq_client:
                st.error("GROQ_API_KEY is not configured.")
            else:
                with st.spinner("Running multimodal creative pipeline..."):
                    try:
                        result = generate_pipeline(
                            prompt=prompt,
                            language=language,
                            tone=tone,
                            creativity=creativity,
                            image=source_image,
                        )
                        meme_id = save_meme(result, prompt, language, tone)
                        result["meme_id"] = meme_id
                        st.session_state.last_result = result
                        st.success(f"Generated in {result['latency_ms']} ms")
                    except Exception as exc:
                        st.error(f"Generation failed: {exc}")

        result = st.session_state.last_result

        if result:
            st.image(result["image"], use_container_width=True)

            caption = result["winner"].caption
            st.markdown(
                f"""
                <div class="card">
                  <div class="tiny">SELECTED CAPTION</div>
                  <div style="font-size:20px;font-weight:800;margin-top:5px;">
                    {caption}
                  </div>
                  <div class="tiny" style="margin-top:10px;">
                    {result["winner"].rationale}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            buf = io.BytesIO()
            result["image"].save(buf, format="PNG")

            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    "Download PNG",
                    buf.getvalue(),
                    file_name="memegen_x.png",
                    mime="image/png",
                    use_container_width=True,
                )
            with d2:
                msg = urllib.parse.quote(
                    f"{caption}\n\nCreated with MemeGen X"
                )
                st.link_button(
                    "Share",
                    f"https://wa.me/?text={msg}",
                    use_container_width=True,
                )
        else:
            st.markdown(
                """
                <div class="card" style="height:520px;display:flex;align-items:center;justify-content:center;text-align:center;">
                  <div>
                    <div style="font-size:70px;">🧠</div>
                    <h2>Creative canvas</h2>
                    <div class="muted">
                      Your ranked AI-generated meme will appear here.
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        st.markdown("### AI Copilot")

        if result:
            score = result["winner_score"]

            st.markdown(
                f"""
                <div class="metric-card">
                  <div class="tiny">OVERALL QUALITY</div>
                  <div class="metric-value">{score.overall*100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            score_items = [
                ("Relevance", score.relevance),
                ("Humor", score.humor),
                ("Language", score.language),
                ("Originality", score.originality),
                ("Visual fit", score.visual),
                ("Safety", score.safety),
            ]

            for label, value in score_items:
                st.markdown(
                    f"""
                    <div style="font-size:13px;margin-top:12px;">
                      <div style="display:flex;justify-content:space-between;">
                        <span>{label}</span><b>{value*100:.0f}%</b>
                      </div>
                      <div class="score-bar">
                        <div class="score-fill" style="width:{value*100:.1f}%"></div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("#### Pipeline")

            for item in [
                "Intent extraction",
                "Candidate generation",
                "Local feature ranking",
                "GPT-OSS quality evaluation",
                "Safety gate",
                "CV layout optimization",
            ]:
                st.markdown(f"✓ {item}")

            st.markdown("#### Candidates")

            for i, (cand, score) in enumerate(
                zip(result["candidates"], result["scores"]), start=1
            ):
                with st.expander(
                    f"{i}. {score.overall*100:.0f}% — {cand.caption[:48]}"
                ):
                    st.write(cand.caption)
                    st.caption(cand.rationale)

        else:
            st.info("Generate a meme to see model reasoning and quality signals.")


# ============================================================================
# VISION
# ============================================================================

elif st.session_state.page == "Vision Roast":
    st.markdown(
        """
        <div class="hero">
          <div class="pill">VISION AI</div>
          <div class="hero-title">Upload an image. Let the <span class="gradient">vision engine</span> understand it.</div>
          <div class="muted">Qwen multimodal analysis + OCR/context extraction + CV layout.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    upload = st.file_uploader(
        "Upload image",
        type=["png", "jpg", "jpeg", "webp"],
        key="vision_upload",
    )

    if upload:
        image = Image.open(upload).convert("RGB")

        a, b = st.columns(2)
        with a:
            st.image(image, use_container_width=True)

        with b:
            instruction = st.text_area(
                "Vision instruction",
                "Analyze this image for meme potential and identify the safest caption zone.",
            )

            if st.button("Analyze with Vision AI", type="primary", use_container_width=True):
                if not groq_client:
                    st.error("GROQ_API_KEY is not configured.")
                else:
                    with st.spinner("Running vision model..."):
                        try:
                            vision = vision_analyze(
                                image_to_bytes(image),
                                instruction,
                            )
                            st.session_state["vision_result"] = vision
                        except Exception as exc:
                            st.error(str(exc))

        vision = st.session_state.get("vision_result")

        if vision:
            st.markdown("### Vision intelligence")
            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Emotion", str(vision.get("emotion", "—")))
            c2.metric("Safe zone", str(vision.get("safe_zone", "—")))
            c3.metric("Text", "Detected" if vision.get("text_in_image") else "None")
            c4.metric("Model", "Qwen 3.8 27B")

            st.json(vision)

            if st.button("🔥 Generate roast from this image", type="primary"):
                prompt = (
                    f"Create a short meme caption for this image. "
                    f"Visual analysis: {json.dumps(vision)}"
                )
                with st.spinner("Creating ranked roast..."):
                    try:
                        result = generate_pipeline(
                            prompt,
                            st.session_state.language,
                            st.session_state.tone,
                            st.session_state.creativity,
                            image,
                        )
                        save_meme(
                            result,
                            prompt,
                            st.session_state.language,
                            st.session_state.tone,
                        )
                        st.session_state.last_result = result
                        st.image(result["image"], use_container_width=True)
                        st.success(result["winner"].caption)
                    except Exception as exc:
                        st.error(str(exc))


# ============================================================================
# VAULT
# ============================================================================

elif st.session_state.page == "Vault":
    st.markdown(
        """
        <div class="hero">
          <div class="pill">MEME VAULT</div>
          <div class="hero-title">Your generated <span class="gradient">creative history.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rows = history()

    if not rows:
        st.info("No generated memes yet.")
    else:
        for row in rows:
            with st.container():
                c1, c2 = st.columns([1.1, 1])

                with c1:
                    path = Path(row["image_path"])
                    if path.exists():
                        st.image(str(path), use_container_width=True)

                with c2:
                    st.markdown(f"### {row['caption']}")
                    st.caption(
                        f"{row['language']} · {row['tone']} · "
                        f"{row['model']} · {row['latency_ms']} ms"
                    )

                    cols = st.columns(3)
                    cols[0].metric("Quality", f"{(row['overall'] or 0)*100:.0f}%")
                    cols[1].metric("Humor", f"{(row['humor'] or 0)*100:.0f}%")
                    cols[2].metric("Relevance", f"{(row['relevance'] or 0)*100:.0f}%")

                st.divider()


# ============================================================================
# ANALYTICS
# ============================================================================

elif st.session_state.page == "Analytics":
    rows = history()

    st.markdown(
        """
        <div class="hero">
          <div class="pill">AI OBSERVABILITY</div>
          <div class="hero-title">Creative system <span class="gradient">analytics.</span></div>
          <div class="muted">Track quality, latency and generation volume.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total = len(rows)
    avg_quality = (
        sum((r["overall"] or 0) for r in rows) / total
        if total else 0
    )
    avg_latency = (
        sum((r["latency_ms"] or 0) for r in rows) / total
        if total else 0
    )

    a, b, c, d = st.columns(4)
    a.metric("Generations", total)
    b.metric("Avg quality", f"{avg_quality*100:.1f}%")
    c.metric("Avg latency", f"{avg_latency:.0f} ms")
    d.metric("AI model", "GPT-OSS 120B")

    if rows:
        st.markdown("### Quality distribution")

        chart = []
        for r in rows:
            chart.append(
                {
                    "quality": float(r["overall"] or 0),
                    "humor": float(r["humor"] or 0),
                    "relevance": float(r["relevance"] or 0),
                    "originality": float(r["originality"] or 0),
                }
            )

        st.bar_chart(chart)


# ============================================================================
# AI LAB
# ============================================================================

elif st.session_state.page == "AI Lab":
    st.markdown(
        """
        <div class="hero">
          <div class="pill">MODEL LAB</div>
          <div class="hero-title">Inspect the <span class="gradient">AI architecture.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Active model registry")

    registry = [
        {
            "role": "Primary reasoning",
            "model": MAIN_MODEL,
            "status": "Production",
            "purpose": "Candidate generation, evaluation, reasoning",
        },
        {
            "role": "Fast router",
            "model": FAST_MODEL,
            "status": "Production",
            "purpose": "Intent classification and lightweight tasks",
        },
        {
            "role": "Vision",
            "model": VISION_MODEL,
            "status": "Preview",
            "purpose": "Image understanding, OCR and visual QA",
        },
        {
            "role": "Safety",
            "model": SAFETY_MODEL,
            "status": "Preview",
            "purpose": "Policy/safety classification",
        },
        {
            "role": "Speech",
            "model": STT_MODEL,
            "status": "Production",
            "purpose": "Voice transcription",
        },
    ]

    st.dataframe(registry, use_container_width=True, hide_index=True)

    st.markdown("### Ranking algorithm")

    st.code(
        """
overall =
    0.30 * relevance
  + 0.25 * humor
  + 0.15 * language
  + 0.15 * originality
  + 0.10 * visual_fit
  + 0.05 * safety

Pipeline:
prompt
  -> intent extraction
  -> 6 candidate generation
  -> deterministic feature extraction
  -> local ML-style ranking
  -> top-4 shortlist
  -> GPT-OSS 120B evaluator
  -> safety gate
  -> computer-vision layout
  -> final image
        """,
        language="text",
    )

    st.info(
        "The local scorer is a transparent heuristic/ranking layer, not a "
        "pretend-trained neural network. For a real enterprise deployment, "
        "replace it with a learned ranker trained on user feedback and evaluation labels."
    )


# ============================================================================
# SETTINGS
# ============================================================================

elif st.session_state.page == "Settings":
    st.markdown(
        """
        <div class="hero">
          <div class="pill">WORKSPACE SETTINGS</div>
          <div class="hero-title">Control your <span class="gradient">AI studio.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Supabase configuration")
    st.code(
        f"""SUPABASE_URL={SUPABASE_URL}
SUPABASE_PUBLISHABLE_KEY=<your sb_publishable_... key>
SUPABASE_SECRET_KEY=<server-only sb_secret_... key>""",
        language="bash",
    )
    st.caption(
        "Use the publishable key for this Streamlit user path with RLS. "
        "Keep the secret key server-only and never commit it."
    )

    st.markdown("### Model configuration")

    st.code(
        f"""
MEMEGEN_MAIN_MODEL={MAIN_MODEL}
MEMEGEN_FAST_MODEL={FAST_MODEL}
MEMEGEN_VISION_MODEL={VISION_MODEL}
MEMEGEN_SAFETY_MODEL={SAFETY_MODEL}
MEMEGEN_STT_MODEL={STT_MODEL}
        """.strip(),
        language="bash",
    )

    st.markdown("### Product controls")

    new_tone = st.selectbox(
        "Default tone",
        ["Savage", "Wholesome", "Sarcastic", "Chaotic", "Corporate", "Dry"],
        index=["Savage", "Wholesome", "Sarcastic", "Chaotic", "Corporate", "Dry"].index(
            st.session_state.tone
        ),
    )

    if st.button("Save defaults"):
        st.session_state.tone = new_tone
        st.success("Workspace defaults updated.")

    st.markdown("### Security")

    st.info(
        "For public deployment, move authentication to Supabase/Auth0/enterprise SSO, "
        "use PostgreSQL with row-level security, store images in object storage, "
        "put API calls behind a backend service, add rate limiting and secrets management."
    )

    st.markdown("### Audit")

    conn = db()
    logs = conn.execute(
        """
        SELECT event, metadata, created_at
        FROM audit_logs
        WHERE user_id=?
        ORDER BY created_at DESC
        LIMIT 50
        """,
        (st.session_state.user_id,),
    ).fetchall()
    conn.close()

    if logs:
        st.dataframe(
            [
                {
                    "event": x["event"],
                    "metadata": x["metadata"],
                    "created_at": x["created_at"],
                }
                for x in logs
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No audit events yet.")
