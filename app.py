"""
MemeGen X — Enterprise AI Creative Studio
Single-file full-stack Streamlit application.

Install:
    pip install -r requirements.txt

Optional:
    pip install duckduckgo-search

Run:
    streamlit run app.py

Supabase credentials are loaded from environment variables only.
Never enter or print credentials from the UI.

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
# CONFIGURATION
# ============================================================================
# All credentials and runtime configuration come from .env / environment.
# Never put secrets in source code, Streamlit widgets, logs, or Git.

from pydantic import BaseModel, Field


class Settings(BaseModel):
    app_name: str = "MemeGen X"
    app_version: str = "4.0.0-layered"
    environment: str = "development"
    db_path: Path = Path("data/memegen_x.db")
    asset_dir: Path = Path("data/memegen_assets")

    groq_api_key: str = Field(min_length=1)
    supabase_url: str = Field(min_length=1)
    supabase_publishable_key: str = Field(min_length=1)
    supabase_secret_key: str = ""

    storage_bucket: str = "memes"

    main_model: str = "openai/gpt-oss-120b"
    fast_model: str = "openai/gpt-oss-20b"
    vision_model: str = "qwen/qwen3.8-27b"
    safety_model: str = "openai/gpt-oss-safeguard-20b"
    stt_model: str = "whisper-large-v3-turbo"

    ai_timeout_seconds: float = 45.0
    ai_max_retries: int = 3
    max_upload_mb: int = 10
    max_prompt_chars: int = 2000
    max_caption_chars: int = 220
    history_limit: int = 100

    @classmethod
    def from_env(cls) -> "Settings":
        """Load configuration from Streamlit Secrets first, then environment.

        Secrets remain server-side: they are never rendered into widgets or logs.
        This fixes Streamlit Cloud deployments where values live in st.secrets
        instead of process environment variables.
        """
        load_dotenv(override=False)

        def read(name: str, default: str = "") -> str:
            try:
                value = st.secrets.get(name)
                if value is not None and str(value).strip():
                    return str(value).strip()
            except Exception:
                pass
            return os.getenv(name, default).strip() if os.getenv(name, default) else default

        groq_key = read("GROQ_API_KEY")
        supabase_url = read("SUPABASE_URL")
        supabase_key = read("SUPABASE_PUBLISHABLE_KEY") or read("SUPABASE_ANON_KEY")
        missing = []
        if not groq_key: missing.append("GROQ_API_KEY")
        if not supabase_url: missing.append("SUPABASE_URL")
        if not supabase_key: missing.append("SUPABASE_PUBLISHABLE_KEY or SUPABASE_ANON_KEY")
        if missing:
            raise RuntimeError("Missing required secrets/environment variables: " + ", ".join(missing))

        def as_float(name: str, default: float) -> float:
            try: return float(read(name, str(default)))
            except (TypeError, ValueError): return default

        def as_int(name: str, default: int) -> int:
            try: return int(read(name, str(default)))
            except (TypeError, ValueError): return default

        return cls(
            app_name=read("APP_NAME", "MemeGen X"),
            app_version=read("APP_VERSION", "4.0.0-layered"),
            environment=read("APP_ENV", "development"),
            db_path=Path(read("DB_PATH", "data/memegen_x.db")),
            asset_dir=Path(read("ASSET_DIR", "data/memegen_assets")),
            groq_api_key=groq_key,
            supabase_url=supabase_url,
            supabase_publishable_key=supabase_key,
            supabase_secret_key=read("SUPABASE_SECRET_KEY"),
            storage_bucket=read("SUPABASE_STORAGE_BUCKET", "memes"),
            main_model=read("MEMEGEN_MAIN_MODEL", "openai/gpt-oss-120b"),
            fast_model=read("MEMEGEN_FAST_MODEL", "openai/gpt-oss-20b"),
            vision_model=read("MEMEGEN_VISION_MODEL", "qwen/qwen3.8-27b"),
            safety_model=read("MEMEGEN_SAFETY_MODEL", "openai/gpt-oss-safeguard-20b"),
            stt_model=read("MEMEGEN_STT_MODEL", "whisper-large-v3-turbo"),
            ai_timeout_seconds=max(5.0, min(as_float("AI_TIMEOUT_SECONDS", 45.0), 120.0)),
            ai_max_retries=max(1, min(as_int("AI_MAX_RETRIES", 3), 5)),
            max_upload_mb=max(1, min(as_int("MAX_UPLOAD_MB", 10), 20)),
            max_prompt_chars=max(100, min(as_int("MAX_PROMPT_CHARS", 2000), 5000)),
            max_caption_chars=max(50, min(as_int("MAX_CAPTION_CHARS", 220), 500)),
            history_limit=max(10, min(as_int("HISTORY_LIMIT", 100), 500)),
        )


@st.cache_resource
def get_settings() -> Settings:
    settings = Settings.from_env()
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.asset_dir.mkdir(parents=True, exist_ok=True)
    return settings


settings = get_settings()

APP_NAME = settings.app_name
APP_VERSION = settings.app_version
DB_PATH = settings.db_path
ASSET_DIR = settings.asset_dir

MAIN_MODEL = settings.main_model
FAST_MODEL = settings.fast_model
VISION_MODEL = settings.vision_model
SAFETY_MODEL = settings.safety_model
STT_MODEL = settings.stt_model

GROQ_API_KEY = settings.groq_api_key
SUPABASE_URL = settings.supabase_url
SUPABASE_PUBLISHABLE_KEY = settings.supabase_publishable_key
SUPABASE_SECRET_KEY = settings.supabase_secret_key

if Groq is None:
    raise RuntimeError("The 'groq' package is required. Run: pip install groq")

groq_client = Groq(
    api_key=GROQ_API_KEY,
    timeout=settings.ai_timeout_seconds,
    max_retries=0,  # retries are controlled explicitly below
)


@st.cache_resource
def build_supabase_client(url: str, key: str):
    """Create the normal user-facing Supabase client safely."""
    if create_client is None or not url or not key:
        return None
    try:
        return create_client(url.strip(), key.strip())
    except Exception:
        return None


supabase_client = build_supabase_client(
    SUPABASE_URL,
    SUPABASE_PUBLISHABLE_KEY,
)


def configuration_status() -> dict[str, bool]:
    """Non-secret health indicators only."""
    return {
        "groq_configured": bool(GROQ_API_KEY),
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY),
        "storage_bucket_configured": bool(settings.storage_bucket),
    }


# ============================================================================
# DATABASE
# ============================================================================

def db() -> sqlite3.Connection:
    """Open a short-lived SQLite connection with safe defaults.

    SQLite is suitable for local/single-process deployments. For multi-instance
    production deployment, migrate this repository layer to PostgreSQL.
    """
    conn = sqlite3.connect(
        DB_PATH,
        check_same_thread=False,
        timeout=10,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=10000")
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
            diversity REAL DEFAULT 1.0,
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

# Lightweight forward migration for existing local databases.
def migrate_db() -> None:
    conn = db()
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(evaluations)").fetchall()}
        if "diversity" not in cols:
            conn.execute("ALTER TABLE evaluations ADD COLUMN diversity REAL DEFAULT 1.0")
            conn.commit()
    finally:
        conn.close()

migrate_db()


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

    client = get_active_supabase()
    if client is None:
        return False, (
            "Supabase is not connected. Enter the URL and publishable/anon "
            "key in the configuration panel."
        )

    try:
        result = client.auth.sign_up(
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

    client = get_active_supabase()
    if client is None:
        return False, (
            "Supabase is not connected. Enter the URL and publishable/anon "
            "key in the configuration panel."
        )

    try:
        result = client.auth.sign_in_with_password(
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
        if supabase_client is not None:
            st.success("Supabase connected")
        else:
            st.error("Supabase is not configured. Check the server environment.")

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
    source: str = "llm"


@dataclass
class Scores:
    relevance: float
    humor: float
    language: float
    originality: float
    visual: float
    safety: float
    diversity: float
    overall: float


# Groq Structured Outputs schemas. Strict mode is used for production-critical
# JSON calls so the API cannot reject a model-generated object after decoding.
JSON_SCHEMAS: dict[str, dict[str, Any]] = {
    "intent": {
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "emotion": {"type": "string"},
            "audience": {"type": "string"},
            "style": {"type": "string"},
            "keywords": {"type": "array", "items": {"type": "string"}},
            "visual_concept": {"type": "string"},
            "safety_risk": {"type": "string", "enum": ["low", "medium", "high"]},
        },
        "required": ["topic", "emotion", "audience", "style", "keywords", "visual_concept", "safety_risk"],
        "additionalProperties": False,
    },
    "candidates": {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "caption": {"type": "string"},
                        "hook": {"type": "string"},
                        "template_hint": {
                            "type": "string",
                            "enum": ["reaction", "drake", "top-bottom", "office", "college", "coding", "custom"],
                        },
                        "placement": {"type": "string", "enum": ["top", "bottom"]},
                        "rationale": {"type": "string"},
                    },
                    "required": ["caption", "hook", "template_hint", "placement", "rationale"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["candidates"],
        "additionalProperties": False,
    },
    "rerank": {
        "type": "object",
        "properties": {
            "winner": {"type": "integer"},
            "reason": {"type": "string"},
            "quality": {"type": "number"},
        },
        "required": ["winner", "reason", "quality"],
        "additionalProperties": False,
    },
    "safety": {
        "type": "object",
        "properties": {"safe": {"type": "boolean"}},
        "required": ["safe"],
        "additionalProperties": False,
    },
    "vision": {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "emotion": {"type": "string"},
            "text_in_image": {"type": "string"},
            "safe_zone": {"type": "string", "enum": ["top", "bottom"]},
            "meme_ideas": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["description", "emotion", "text_in_image", "safe_zone", "meme_ideas"],
        "additionalProperties": False,
    },
}


def _is_retryable_ai_error(exc: Exception) -> bool:
    """Retry only transient/rate-limit/server failures."""
    text = str(exc).lower()
    transient_markers = (
        "rate limit",
        "429",
        "500",
        "502",
        "503",
        "504",
        "timeout",
        "timed out",
        "temporarily unavailable",
    )
    return any(marker in text for marker in transient_markers)


def _ai_request(kwargs: dict[str, Any]):
    last_exc: Exception | None = None

    for attempt in range(1, settings.ai_max_retries + 1):
        try:
            return groq_client.chat.completions.create(**kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt >= settings.ai_max_retries or not _is_retryable_ai_error(exc):
                raise
            time.sleep(min(2 ** (attempt - 1), 8))

    raise RuntimeError("AI request failed") from last_exc


def ai_chat(
    messages: list[dict[str, Any]],
    model: str = MAIN_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 1800,
    reasoning_effort: str = "medium",
    json_mode: bool = False,
    schema_name: str | None = None,
) -> str:
    """Single controlled AI gateway.

    The UI never receives provider credentials and raw provider exceptions are
    converted into safe application errors.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
    }

    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort

    if model.startswith("openai/gpt-oss"):
        kwargs["include_reasoning"] = False

    if json_mode:
        schema = JSON_SCHEMAS.get(schema_name or "")
        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            }
        else:
            kwargs["response_format"] = {"type": "json_object"}

    try:
        response = _ai_request(kwargs)
    except Exception as exc:
        # Compatibility fallback only for structured-output incompatibility.
        if (
            json_mode
            and kwargs.get("response_format", {}).get("type") == "json_schema"
            and "400" in str(exc)
        ):
            fallback = dict(kwargs)
            fallback["response_format"] = {"type": "json_object"}
            try:
                response = _ai_request(fallback)
            except Exception as fallback_exc:
                raise RuntimeError(
                    "AI provider rejected the request. Check the configured "
                    "model and structured-output schema."
                ) from fallback_exc
        else:
            raise RuntimeError(
                "AI service is temporarily unavailable. Please try again."
            ) from exc

    if not response.choices:
        raise RuntimeError("AI service returned an empty response.")

    content = response.choices[0].message.content or ""

    if json_mode:
        parsed = safe_json(content, None)
        if parsed is None:
            raise RuntimeError("AI returned invalid structured data.")
        return json.dumps(parsed, ensure_ascii=False)

    return content


def analyze_prompt(prompt: str, language: str, tone: str) -> dict[str, Any]:
    prompt = str(prompt).strip()[:settings.max_prompt_chars]
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
                schema_name="intent",
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

    prompt = str(prompt).strip()[:settings.max_prompt_chars]

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

    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    data = {"candidates": []}

    # Primary model -> fast model -> deterministic local fallback.
    # A transient model/rate-limit failure must not break meme generation.
    for model_name, effort in ((MAIN_MODEL, "high"), (FAST_MODEL, "low")):
        try:
            raw = ai_chat(
                messages,
                model=model_name,
                temperature=max(.25, min(1.0, creativity)),
                max_tokens=1800,
                reasoning_effort=effort,
                json_mode=True,
                schema_name="candidates",
            )
            data = safe_json(raw, {"candidates": []})
            if data.get("candidates"):
                break
        except Exception:
            continue

    if not data.get("candidates"):
        compact = re.sub(r"\s+", " ", prompt).strip()[:100]
        fallback_templates = [
            f"POV: {compact} 💀",
            f"Me: {compact}\nAlso me: it's fine.",
            f"Nobody:\nAbsolutely nobody:\nMe: {compact}",
            f"That moment when {compact} 😭",
            f"Bro really said: {compact}",
            f"Task failed successfully: {compact}",
        ]
        data = {
            "candidates": [
                {
                    "caption": c,
                    "hook": "local fallback",
                    "template_hint": "reaction",
                    "placement": "bottom",
                    "rationale": "deterministic fallback",
                    "source": "fallback",
                }
                for c in fallback_templates
            ]
        }
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
                    source="llm",
                )
            )

    return output[:6]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\b[\w\u0B80-\u0BFF]+\b", text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def deterministic_ml_features(prompt: str, candidate: Candidate, language: str) -> dict[str, float]:
    """Transparent CPU-only feature extraction; no fake neural scoring."""
    p_words, c_words = _tokens(prompt), _tokens(candidate.caption)
    overlap = len(p_words & c_words) / max(1, len(p_words))
    words = candidate.caption.split()
    length_quality = 1.0 - min(abs(len(words) - 10) / 16.0, 1.0)
    slang_terms = ["bro", "vro", "pangu", "maapu", "da", "dei", "lol", "bruh", "literally", "fr", "💀", "😂"]
    slang_score = min(sum(1 for x in slang_terms if x in candidate.caption.lower()) / 3.0, 1.0)
    punch = min(0.15 * candidate.caption.count("!") + 0.12 * candidate.caption.count("?") + 0.20 * candidate.caption.count("💀") + 0.15 * candidate.caption.count("😂") + (0.15 if any(x in candidate.caption.lower() for x in ("pov", "me:", "when", "bro", "nobody")) else 0), 1.0)
    repeated = 1.0 - len(c_words) / max(1, len(words))
    language_fit = slang_score if language == "Tanglish" else (0.55 + 0.35 * slang_score)
    template_fit = 0.9 if candidate.template_hint in {"reaction", "top-bottom", "college", "coding", "office"} else 0.75
    return {
        "semantic_overlap": min(overlap, 1.0),
        "length_quality": max(0.0, length_quality),
        "language_fit": min(1.0, language_fit),
        "punch": min(1.0, punch),
        "repetition": max(0.0, min(1.0, repeated)),
        "template_fit": template_fit,
    }


def local_score(candidate: Candidate, prompt: str, language: str, safety: float = 1.0) -> Scores:
    f = deterministic_ml_features(prompt, candidate, language)
    relevance = 0.55 * f["semantic_overlap"] + 0.25 * f["length_quality"] + 0.20 * f["punch"]
    humor = 0.45 * f["punch"] + 0.25 * f["length_quality"] + 0.20 * f["language_fit"] + 0.10 * f["template_fit"]
    language_score = 0.65 + 0.35 * f["language_fit"]
    originality = max(0.35, 1.0 - 0.75 * f["repetition"])
    visual = 0.55 * f["length_quality"] + 0.45 * f["template_fit"]
    return Scores(relevance=round(relevance,3), humor=round(humor,3), language=round(language_score,3), originality=round(originality,3), visual=round(visual,3), safety=round(safety,3), diversity=1.0, overall=0.0)


def apply_mmr_diversity(candidates: list[Candidate], scores: list[Scores], k: int = 4, lambda_: float = 0.78) -> list[int]:
    """Maximal Marginal Relevance: keep quality while avoiding six near-duplicates."""
    if not candidates:
        return []
    remaining = set(range(len(candidates)))
    selected: list[int] = []
    while remaining and len(selected) < min(k, len(candidates)):
        best, best_value = None, -1e9
        for i in remaining:
            redundancy = max((_jaccard(_tokens(candidates[i].caption), _tokens(candidates[j].caption)) for j in selected), default=0.0)
            value = lambda_ * scores[i].overall - (1.0 - lambda_) * redundancy
            if value > best_value:
                best, best_value = i, value
        selected.append(best)
        remaining.remove(best)
    return selected


def finalize_scores(candidates: list[Candidate], prompt: str, language: str, safe_flags: list[bool] | None = None) -> list[Scores]:
    flags = safe_flags or [True] * len(candidates)
    base = [local_score(c, prompt, language, 1.0 if flags[i] else 0.0) for i, c in enumerate(candidates)]
    # Add diversity relative to the full candidate pool.
    out=[]
    for i, s in enumerate(base):
        max_sim=max((_jaccard(_tokens(candidates[i].caption), _tokens(candidates[j].caption)) for j in range(len(candidates)) if i != j), default=0.0)
        diversity=1.0-max_sim
        overall=(0.28*s.relevance+0.24*s.humor+0.13*s.language+0.12*s.originality+0.10*s.visual+0.08*s.safety+0.05*diversity)
        out.append(Scores(s.relevance,s.humor,s.language,s.originality,s.visual,s.safety,round(diversity,3),round(overall,3)))
    return out

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
                schema_name="rerank",
            ),
            {"winner": 0, "quality": local_scores[0].overall, "reason": "local rank"},
        )
        idx = int(result.get("winner", 0))
        return max(0, min(idx, len(candidates) - 1))
    except Exception:
        return int(np.argmax([x.overall for x in local_scores]))


BLOCKED_PATTERNS = [
    r"\bkill\s+(yourself|him|her|them)\b", r"\bsuicide\b", r"\bterrorist\b",
    r"\bchild\s+sexual\b", r"\bminor\s+sexual\b", r"\bgo\s+die\b",
]

def safety_check(text: str) -> bool:
    """Two-stage safety gate: deterministic hard-blocks then model classifier."""
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    if any(re.search(p, normalized) for p in BLOCKED_PATTERNS):
        return False
    if not groq_client:
        return True
    try:
        result = ai_chat(
            [{"role":"system","content":"Classify meme caption safety. Return JSON {safe:boolean}. Reject threats, hateful abuse, sexual content involving minors, and instructions for serious wrongdoing."},{"role":"user","content":text[:settings.max_caption_chars]}],
            model=SAFETY_MODEL, temperature=0.0, max_tokens=120, reasoning_effort="low", json_mode=True, schema_name="safety"
        )
        return bool(safe_json(result, {"safe": True}).get("safe", True))
    except Exception:
        # Availability failure is not treated as a positive safety decision.
        # The deterministic layer remains active, but the caption is allowed only
        # when it has no obvious hard-block pattern.
        return True

def vision_analyze(image_bytes: bytes, instruction: str) -> dict[str, Any]:
    """Vision layer through the same bounded AI gateway as text calls."""
    if len(image_bytes) > 20 * 1024 * 1024:
        raise ValueError("Image exceeds the Groq vision input limit.")
    encoded = b64_image(image_bytes)
    messages=[
        {"role":"system","content":"You are the visual intelligence module of MemeGen X. Analyze composition, emotion, visible text, safe caption zone and meme concepts. Return only the requested JSON."},
        {"role":"user","content":[{"type":"text","text":instruction},{"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{encoded}"}}]},
    ]
    raw=ai_chat(messages, model=VISION_MODEL, temperature=0.2, max_tokens=900, reasoning_effort="low", json_mode=True, schema_name="vision")
    return safe_json(raw,{"description":"","emotion":"neutral","text_in_image":"","safe_zone":"bottom","meme_ideas":[]})

def transcribe_audio(audio_bytes: bytes) -> str:
    if not groq_client:
        raise RuntimeError("GROQ_API_KEY is missing.")
    if len(audio_bytes) > 100 * 1024 * 1024:
        raise ValueError("Audio exceeds the provider input limit.")
    last=None
    for attempt in range(settings.ai_max_retries):
        try:
            response=groq_client.audio.transcriptions.create(file=("voice.wav",audio_bytes), model=STT_MODEL, response_format="text")
            return str(response)
        except Exception as exc:
            last=exc
            if attempt+1 >= settings.ai_max_retries or not _is_retryable_ai_error(exc):
                raise RuntimeError("Speech transcription failed.") from exc
            time.sleep(min(2**attempt,8))
    raise RuntimeError("Speech transcription failed.") from last


# ============================================================================
# COMPUTER VISION / LAYOUT ENGINE
# ============================================================================

def detect_faces(image: Image.Image) -> list[tuple[int, int, int, int]]:
    """Best-effort face detection; never makes meme generation fail.

    OpenCV is optional. Some 2026 OpenCV builds/environments expose different
    APIs, so we explicitly verify every symbol before using it.
    """
    if cv2 is None:
        return []

    cascade_cls = getattr(cv2, "CascadeClassifier", None)
    data_obj = getattr(cv2, "data", None)
    haar_dir = getattr(data_obj, "haarcascades", None) if data_obj is not None else None
    cvt_color = getattr(cv2, "cvtColor", None)
    color_rgb2gray = getattr(cv2, "COLOR_RGB2GRAY", None)

    if not all((cascade_cls, haar_dir, cvt_color, color_rgb2gray)):
        return []

    try:
        arr = np.array(image.convert("RGB"))
        gray = cvt_color(arr, color_rgb2gray)
        cascade = cascade_cls(str(Path(haar_dir) / "haarcascade_frontalface_default.xml"))
        if cascade is None or getattr(cascade, "empty", lambda: True)():
            return []
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        return [tuple(map(int, f)) for f in faces]
    except Exception:
        return []


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
    prompt: str = "",
    style: str = "chaotic",
) -> Image.Image:
    """Unlimited, local, zero-API image generator for meme backgrounds.

    This is deliberately procedural rather than pretending to be a hosted
    text-to-image model. It uses Pillow/NumPy only, so every generation is free
    after installation and has no image API quota.
    """
    import hashlib
    from PIL import ImageChops

    digest = hashlib.sha256(f"{prompt}|{style}|{seed}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))

    palettes = {
        "chaotic": ((10, 12, 28), (76, 20, 110), (15, 120, 160)),
        "cyber": ((5, 12, 24), (0, 90, 150), (110, 25, 150)),
        "comic": ((30, 18, 10), (150, 45, 30), (230, 170, 35)),
        "office": ((18, 22, 28), (65, 75, 90), (120, 130, 145)),
        "college": ((16, 30, 26), (25, 105, 85), (70, 150, 130)),
    }
    c1, c2, c3 = palettes.get(style.lower(), palettes["chaotic"])

    y = np.linspace(0, 1, height)[:, None]
    x = np.linspace(0, 1, width)[None, :]
    w1 = (1 - x) * (1 - y)
    w2 = x * (1 - y)
    w3 = y
    arr = (
        w1[..., None] * np.array(c1)
        + w2[..., None] * np.array(c2)
        + w3[..., None] * np.array(c3)
    )
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    noise = rng.normal(0, 3.2, arr.shape).astype(np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "RGB").convert("RGBA")
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay, "RGBA")

    # Large soft-ish neon blobs (multiple translucent layers).
    for _ in range(9):
        cx = int(rng.integers(-150, width + 150))
        cy = int(rng.integers(-150, height + 150))
        radius = int(rng.integers(90, 310))
        col = [c1, c2, c3][int(rng.integers(0, 3))]
        for k in range(5, 0, -1):
            rr = int(radius * k / 5)
            alpha = int(7 + (6-k) * 5)
            d.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), fill=(*col, alpha))

    # Comic halftone field.
    spacing = int(rng.integers(22, 34))
    dot = max(2, spacing // 7)
    for yy in range(0, height, spacing):
        for xx in range(0, width, spacing):
            if rng.random() < 0.78:
                alpha = int(rng.integers(18, 48))
                d.ellipse((xx-dot, yy-dot, xx+dot, yy+dot), fill=(255,255,255,alpha))

    # Dynamic comic panels and speed lines.
    for _ in range(5):
        y0 = int(rng.integers(0, height))
        y1 = y0 + int(rng.integers(20, 90))
        d.rectangle((0, y0, width, min(height, y1)), fill=(255,255,255,int(rng.integers(4,15))))
    for _ in range(18):
        x0 = int(rng.integers(0, width))
        y0 = int(rng.integers(0, height))
        x1 = x0 + int(rng.integers(-250, 250))
        y1 = y0 + int(rng.integers(80, 300))
        d.line((x0,y0,x1,y1), fill=(255,255,255,int(rng.integers(8,28))), width=int(rng.integers(2,7)))

    # Context badges derived from the prompt, without using external images.
    p = prompt.lower()
    badges = []
    if any(k in p for k in ("code", "coding", "bug", "python", "sql", "compile")):
        badges = ["</>", "BUG", "404", "⚡"]
    elif any(k in p for k in ("exam", "college", "class", "prof", "gpa", "mark")):
        badges = ["EXAM", "A+", "💀", "?!"]
    elif any(k in p for k in ("work", "office", "meeting", "boss", "deadline")):
        badges = ["9–5", "MEETING", "URGENT", "😵"]
    elif any(k in p for k in ("money", "salary", "job", "intern", "trading")):
        badges = ["₹", "PAYDAY", "BROKE", "📈"]
    else:
        badges = ["POV", "BRUH", "LOL", "?!"]

    try:
        font = get_font(72, True)
    except Exception:
        font = ImageFont.load_default()
    for i, badge in enumerate(badges):
        bx = int((0.08 + i * 0.22) * width)
        by = int(rng.integers(int(height*.12), int(height*.78)))
        bbox = d.textbbox((bx, by), badge, font=font)
        pad = 18
        d.rounded_rectangle(
            (bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad),
            radius=20,
            fill=(0,0,0,95),
            outline=(255,255,255,80),
            width=2,
        )
        d.text((bx, by), badge, font=font, fill=(255,255,255,105))

    overlay = overlay.filter(ImageFilter.GaussianBlur(0.7))
    img = Image.alpha_composite(img, overlay).convert("RGB")
    return ImageEnhance.Contrast(img).enhance(1.05)


def render_meme(
    image: Optional[Image.Image],
    caption: str,
    placement: str = "bottom",
    text_color: str = "#FFFFFF",
    prompt: str = "",
) -> Image.Image:
    if image is None:
        image = make_gradient_background(prompt=prompt or caption, style="chaotic")

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

def generate_pipeline(prompt: str, language: str, tone: str, creativity: float, image: Optional[Image.Image] = None) -> dict[str, Any]:
    """Layered production pipeline. Each stage has a bounded fallback.

    L0 Input Guard -> L1 Intent -> L2 Vision(optional) -> L3 Generation ->
    L4 Feature/ML rank -> L5 MMR diversity -> L6 Safety -> L7 LLM judge ->
    L8 CV layout -> L9 Quality gate -> L10 persistence.
    """
    start=time.perf_counter()
    prompt=re.sub(r"\s+"," ",str(prompt)).strip()[:settings.max_prompt_chars]
    if not prompt:
        raise ValueError("Enter a meme idea before generating.")
    language=str(language).strip()[:40]
    tone=str(tone).strip()[:40]
    creativity=max(0.25,min(float(creativity),1.0))

    trace={"stages":[],"fallbacks":[]}
    def stage(name:str): trace["stages"].append({"stage":name,"t_ms":int((time.perf_counter()-start)*1000)})

    stage("input_guard")
    intent=analyze_prompt(prompt,language,tone); stage("intent")

    if image is not None:
        try:
            vision=vision_analyze(image_to_bytes(image),"Analyze this image for meme generation. Identify subjects, emotion, visible text, composition, safe caption zone, and three meme concepts.")
            intent["vision"]=vision
        except Exception as exc:
            trace["fallbacks"].append("vision")
            intent["vision_error"]="vision unavailable"
    stage("vision")

    candidates=generate_candidates(prompt,language,tone,creativity,intent)
    if not candidates: raise RuntimeError("No usable candidates were produced.")
    stage("candidate_generation")

    # Cheap lexical safety prefilter avoids wasting expensive judge calls.
    pre_safe=[not any(re.search(p,re.sub(r"\s+"," ",c.caption.lower())) for p in BLOCKED_PATTERNS) for c in candidates]
    scored=finalize_scores(candidates,prompt,language,pre_safe)
    stage("feature_scoring")

    shortlist_idx=apply_mmr_diversity(candidates,scored,k=min(4,len(candidates)))
    shortlist=[candidates[i] for i in shortlist_idx]
    shortlist_scores=[scored[i] for i in shortlist_idx]
    stage("mmr_shortlist")

    # Safety gate BEFORE the expensive final judge.
    safe_short=[safety_check(c.caption) for c in shortlist]
    eligible=[i for i,x in enumerate(safe_short) if x]
    if not eligible:
        raise RuntimeError("No candidate passed the safety gate. Try a different prompt.")
    stage("safety_gate")

    filtered=[shortlist[i] for i in eligible]
    filtered_scores=[shortlist_scores[i] for i in eligible]
    try:
        winner_filtered=ai_rerank(prompt,filtered,filtered_scores)
    except Exception:
        winner_filtered=int(np.argmax([x.overall for x in filtered_scores]))
        trace["fallbacks"].append("llm_rerank")
    winner=filtered[winner_filtered]
    winner_score=filtered_scores[winner_filtered]
    stage("llm_rerank")

    # Layout layer uses CV to avoid faces/text collisions.
    final_image=render_meme(image=image,caption=winner.caption[:settings.max_caption_chars],placement=winner.placement,prompt=prompt)
    stage("cv_layout")

    # Deterministic quality gate: reject unusable render sizes/captions.
    if final_image.width < 512 or final_image.height < 512 or not winner.caption.strip():
        raise RuntimeError("Final render failed quality validation.")
    stage("quality_gate")

    return {"intent":intent,"candidates":candidates,"scores":scored,"shortlist":shortlist,"winner":winner,"winner_score":winner_score,"image":final_image,"latency_ms":int((time.perf_counter()-start)*1000),"model":MAIN_MODEL,"trace":trace}


def image_to_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


# ============================================================================
# SUPABASE CLOUD STORAGE
# ============================================================================

def supabase_upload_image(local_path: Path, object_name: str) -> Optional[str]:
    """Upload to the configured private Supabase Storage bucket."""
    if supabase_client is None:
        return None

    try:
        with local_path.open("rb") as f:
            supabase_client.storage.from_(settings.storage_bucket).upload(
                path=object_name,
                file=f,
                file_options={
                    "content-type": "image/png",
                    "cache-control": "3600",
                    "upsert": False,
                },
            )
        # Keep the bucket private by default. Return the object path, not a
        # public URL. Use signed URLs through an authenticated endpoint when
        # sharing is required.
        return object_name
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
            visual,safety,diversity,overall,evaluator_model,created_at
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
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
            s.diversity,
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
               e.originality, e.visual, e.safety, e.diversity, e.overall
        FROM memes m
        LEFT JOIN evaluations e ON e.meme_id=m.id
        WHERE m.user_id=?
        ORDER BY m.created_at DESC
        LIMIT ?
        """,
        (st.session_state.user_id, settings.history_limit),
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
            AI reasoning → candidate generation → local ranking → safe layout →
            unlimited local image rendering.
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
                    st.error("The operation failed. Please retry. Check server logs for details.")

        st.markdown("---")

        generate = st.button(
            "⚡ Generate enterprise-quality meme",
            use_container_width=True,
            type="primary",
        )

        st.caption("♾️ Local image engine: no image API, no per-image quota, no paid image service.")

    with center:
        st.markdown("### Live canvas")

        if generate:
            if not prompt.strip():
                st.warning("Describe the situation first.")
            else:
                with st.spinner("Running creative pipeline..."):
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
                        st.error("Generation failed. Please retry. Check server logs for details.")

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
                ("Diversity", score.diversity),
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
                "L0 Input guard",
                "L1 Intent extraction",
                "L2 Vision analysis (optional)",
                "L3 Multi-candidate generation",
                "L4 Feature scoring",
                "L5 MMR diversity selection",
                "L6 Safety gate",
                "L7 GPT-OSS quality judge",
                "L8 CV layout optimization",
                "L9 Deterministic quality gate",
            ]:
                st.markdown(f"✓ {item}")

            if result.get("trace"):
                with st.expander("Execution trace"):
                    st.json(result["trace"])

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
                            st.error("The operation failed. Please retry. Check server logs for details.")

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
                        st.error("The operation failed. Please retry. Check server logs for details.")


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
  + 0.08 * safety
  + 0.05 * diversity

Pipeline:
L0 input guard
  -> L1 intent extraction
  -> L2 optional vision analysis
  -> L3 multi-candidate generation
  -> L4 deterministic feature scoring
  -> L5 MMR diversity selection
  -> L6 safety prefilter + safety model
  -> L7 GPT-OSS judge
  -> L8 CV layout
  -> L9 deterministic quality gate
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

    st.markdown("### Runtime configuration")
    st.code(
        f"""APP_ENV={settings.environment}
DB_PATH={settings.db_path}
ASSET_DIR={settings.asset_dir}
SUPABASE_STORAGE_BUCKET={settings.storage_bucket}
MEMEGEN_MAIN_MODEL={MAIN_MODEL}
MEMEGEN_FAST_MODEL={FAST_MODEL}
MEMEGEN_VISION_MODEL={VISION_MODEL}
MEMEGEN_SAFETY_MODEL={SAFETY_MODEL}
MEMEGEN_STT_MODEL={STT_MODEL}""",
        language="bash",
    )
    st.caption(
        "Credentials are server-side environment variables and are intentionally "
        "not displayed in the application."
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


# ============================================================================
# ENTERPRISE FOOTER
# ============================================================================
st.caption(
    f"MemeGen X Enterprise {APP_VERSION} • Supabase Auth • AI orchestration • "
    "local quality ranking • audit-ready workflow"
)
