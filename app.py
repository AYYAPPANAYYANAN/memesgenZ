import streamlit as st
import requests
import json
import os
import io
import textwrap
import random
import urllib.parse
import base64
import time
from PIL import Image, ImageDraw, ImageFont
from groq import Groq
from duckduckgo_search import DDGS
import cv2
import numpy as np

# --- 1. PAGE CONFIG (must be first Streamlit call) ---
st.set_page_config(page_title="𝔪𝔢𝔪𝔢𝔰 𝔤𝔢𝔫𝔷", page_icon="🧙🏻‍♂", layout="wide")
# --- 2. INITIALIZE ALL SESSION STATES ---

# 🔐 Auth States
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "user_role" not in st.session_state:
    st.session_state.user_role = "user"  # <--- THIS IS THE FIX! Defaults everyone to a standard user.

# 🎨 Meme App States
if "app_theme" not in st.session_state:
    st.session_state.app_theme = "Cyber Blue"
if "meme_history" not in st.session_state:
    st.session_state.meme_history = []
if "generated_caption" not in st.session_state:
    st.session_state.generated_caption = None
if "draft_text" not in st.session_state:
    st.session_state.draft_text = ""



# --- 2.5 PROFESSIONAL SUPABASE AUTHENTICATION ---
from supabase import create_client, Client
import time

# 1. Initialize Auth Session States Safely
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "user_role" not in st.session_state:
    st.session_state.user_role = "user"

# 2. Robust Client Initialization
@st.cache_resource
def get_supabase_client() -> Client:
    """Initializes and caches the Supabase client."""
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except KeyError as e:
        st.error(f"🚨 Missing Secret: {e}. Check your .streamlit/secrets.toml file.")
        st.stop()
    except Exception as e:
        st.error(f"🚨 Failed to initialize Supabase: {e}")
        st.stop()

def render_auth_page():
    """Renders a production-ready authentication UI."""
    st.markdown("<h2 style='text-align: center;'>⚡ Access the Vault</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888;'>Authenticate to continue.</p>", unsafe_allow_html=True)
    
    supabase = get_supabase_client()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container(border=True):
            tab_login, tab_signup = st.tabs(["🔓 Log In", "📝 Sign Up"])
            
            # --- LOGIN TAB ---
            with tab_login:
                with st.form("pro_login_form"):
                    email = st.text_input("Email", placeholder="you@domain.com")
                    password = st.text_input("Password", type="password", placeholder="••••••••")
                    submitted = st.form_submit_button("Log In", use_container_width=True)
                    
                    if submitted:
                        if not email or not password:
                            st.warning("⚠️ Please fill in both fields.")
                        else:
                            with st.spinner("Authenticating securely..."):
                                try:
                                    res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                                    st.session_state.logged_in = True
                                    st.session_state.user_email = res.user.email
                                    
                                    # 👑 THE ADMIN HACK IS PROPERLY PLACED HERE!
                                    # Change this email to your actual login email to get Admin access
                                    if res.user.email == "ayyan@example.com":  
                                        st.session_state.user_role = "admin"
                                    else:
                                        st.session_state.user_role = "user"
                                        
                                    st.success("✅ Access granted! Redirecting...")
                                    time.sleep(1) # Smooth transition
                                    st.rerun()
                                except Exception as e:
                                    error_msg = str(e).lower()
                                    if "timeout" in error_msg or "ssl" in error_msg:
                                        st.error("🚨 Network Timeout: Your internet or antivirus is blocking Supabase. Try a mobile hotspot!")
                                    elif "invalid credentials" in error_msg:
                                        st.error("🚫 Invalid email or password.")
                                    else:
                                        st.error(f"🚨 Auth Error: {e}")

            # --- SIGN UP TAB ---
            with tab_signup:
                with st.form("pro_signup_form"):
                    new_email = st.text_input("Email", placeholder="you@domain.com")
                    new_password = st.text_input("Password", type="password", placeholder="Min 6 characters")
                    signup_submitted = st.form_submit_button("Create Account", use_container_width=True)
                    
                    if signup_submitted:
                        if len(new_password) < 6:
                            st.warning("⚠️ Password must be at least 6 characters.")
                        elif not new_email:
                            st.warning("⚠️ Email is required.")
                        else:
                            with st.spinner("Provisioning account..."):
                                try:
                                    res = supabase.auth.sign_up({"email": new_email, "password": new_password})
                                    st.success("✅ Account created! You can now log in.")
                                except Exception as e:
                                    error_msg = str(e).lower()
                                    if "timeout" in error_msg or "ssl" in error_msg:
                                        st.error("🚨 Network Timeout: Your internet or antivirus is blocking Supabase.")
                                    elif "already registered" in error_msg:
                                        st.warning("⚠️ Email already exists. Try logging in.")
                                    else:
                                        st.error(f"🚨 Registration Error: {e}")

# 3. The Security Gate
if not st.session_state.logged_in:
    render_auth_page()
    st.stop() # Prevents the rest of the app from running until authenticated
# --- 3. DYNAMIC GEN Z THEME ENGINE ---
theme_colors = {
    "Neon Green (Default)": {"primary": "#ccff00", "secondary": "#00ffa3", "bg": "#160b24"},
    "Cyber Blue":           {"primary": "#00f0ff", "secondary": "#0057ff", "bg": "#0a0b14"},
    "Vaporwave Pink":       {"primary": "#ff00ff", "secondary": "#00ffff", "bg": "#1a0b2e"},
    "Sunset Orange":        {"primary": "#ff4d00", "secondary": "#ffcc00", "bg": "#1c0d02"},
}

colors = theme_colors[st.session_state.app_theme]
p_color, s_color, bg_color = colors["primary"], colors["secondary"], colors["bg"]

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800;900&display=swap');

    .stApp {{
        background: radial-gradient(circle at 10% 10%, {bg_color} 0%, #050505 100%);
        color: #fafafa;
        font-family: 'Outfit', sans-serif;
    }}
    h1, h2, h3 {{
        font-family: 'Outfit', sans-serif !important;
        font-weight: 900 !important;
        background: linear-gradient(90deg, {p_color}, {s_color});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 0px 15px {p_color}66;
    }}
    .stButton>button {{
        background: linear-gradient(135deg, {p_color}, {s_color});
        color: #000 !important;
        font-weight: 800;
        border-radius: 12px;
        border: none;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        box-shadow: 0 4px 15px {p_color}4D;
    }}
    .stButton>button:hover {{
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 10px 25px {p_color}99;
    }}
    section[data-testid="stSidebar"] {{
        background-color: rgba(10, 11, 20, 0.6) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 5px 0 30px {p_color}33 !important;
    }}
    div[data-testid="stExpander"] {{
        background: rgba(0, 0, 0, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5) !important;
        transition: all 0.3s ease;
        margin-bottom: 10px;
    }}
    div[data-testid="stExpander"]:hover {{
        border: 1px solid {p_color}80 !important;
        box-shadow: 0 0 20px {p_color}4D !important;
    }}
    .meme-card {{
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }}
    .meme-text {{
        font-size: 24px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, {p_color}, {s_color});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    </style>
""", unsafe_allow_html=True)

# --- 4. GROQ CLIENT SETUP ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=GROQ_API_KEY)
except Exception:
    st.error("🚨 Missing Groq API Key! Check .streamlit/secrets.toml")
    st.stop()

# --- 5. TOP NAVIGATION ---
st.markdown("<h1 style='text-align: center;'>🧙🏻‍♂ 𝔪𝔢𝔪𝔢𝔰 𝔤𝔢𝔫𝔷</h1>", unsafe_allow_html=True)

# --- 6. SIDEBAR SETTINGS ---
with st.sidebar:
    # 1. Update greeting to use user_email
    st.markdown(f"<h3 style='text-align: center;'>👋 Sup, {st.session_state.user_email}</h3>", unsafe_allow_html=True)
    
    # 2. Update logout button to clear user_email
    if st.button("🚪 Log Out", use_container_width=True):
        supabase = get_supabase_client() 
        supabase.auth.sign_out() 
        st.session_state.logged_in = False
        st.session_state.user_email = None  # <-- Make sure this says user_email!
        st.rerun()
        
    st.markdown("---")
    st.markdown("<h2 style='text-align: center;'>⚙️ Workspace</h2>", unsafe_allow_html=True)
    
    # 👑 ADMIN ONLY SECTION
    if st.session_state.user_role == "admin":
        with st.expander("👑 Admin Dashboard", expanded=True):
            st.warning("You have Admin privileges.")
            st.metric("Total Memes Generated (Mock)", "420")
            if st.button("Nuke Meme Vault (Global)"):
                st.success("Vault Cleared!")
    
    # Standard User Settings
    with st.expander("✨ App Appearance", expanded=True):
        selected_theme = st.selectbox(
            "UI Color Theme",
            list(theme_colors.keys()),
            index=list(theme_colors.keys()).index(st.session_state.app_theme),
        )
        if selected_theme != st.session_state.app_theme:
            st.session_state.app_theme = selected_theme
            st.rerun()

    with st.expander("🎨 Graphics Preferences"):
        meme_language = st.selectbox("🌐 Meme Language", ["Tanglish", "English", "German", "Hindi"])
        meme_text_color = st.color_picker("Text Color", "#FFFFFF")
# --- 7. UTILS ---

def burn_meme_text(img: Image.Image, text: str, color: str, position="bottom") -> Image.Image:
    """Burn meme caption text onto a PIL image dynamically dodging faces."""
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size
    font_size = max(20, int(img_h / 10))
    
    try:
        font = ImageFont.truetype("impact.ttf", font_size) # Pro-tip: Try changing arial.ttf to impact.ttf if you have it!
    except IOError:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
            
    line_height = font_size + 10 if hasattr(font, 'getbbox') else 15

    chars_per_line = max(10, int(img_w / (font_size * 0.6)))
    lines = textwrap.wrap(text, width=chars_per_line)
    total_text_height = line_height * len(lines)
    
    # 🔥 THE UPGRADE: Dynamic Y-Axis Placement
    if position == "bottom":
        y_text = img_h - total_text_height - 20
        if y_text < 0: y_text = 10
    else: # "top"
        y_text = 20

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_w = bbox[2] - bbox[0]
        x_text = (img_w - text_w) / 2
        outline_range = max(1, int(font_size / 15))
        
        # Draw the thick black meme outline
        for adj in range(-outline_range, outline_range + 1):
            for op in range(-outline_range, outline_range + 1):
                draw.text((x_text+adj, y_text+op), line, font=font, fill="black")
                
        # Draw the main text color
        draw.text((x_text, y_text), line, font=font, fill=color)
        y_text += line_height
    return img

from duckduckgo_search import DDGS
import json
import random

def internet_meme_agent(situation: str, language: str) -> dict | None:
    """An AI Agent with Graceful Degradation to handle 403 Rate Limits."""
    ddgs = DDGS()
    
    # 🚨 THE EMERGENCY TAMIL VAULT (Used if DuckDuckGo blocks us)
    # These are highly reliable direct Imgflip links to Kollywood legends
    tamil_vault = {
        "Vadivelu Confused/Thinking": "https://i.imgflip.com/49mcdq.jpg",
        "Vadivelu Pain/Crying": "https://i.imgflip.com/38w1b8.jpg",
        "Vadivelu Nesamani/Shock": "https://i.imgflip.com/32688u.jpg",
        "Santhanam Mocking/Sarcasm": "https://i.imgflip.com/2xtw9m.jpg",
        "Goundamani Angry/Tired": "https://i.imgflip.com/5zbncx.jpg",
        "Drake (Fallback)": "https://i.imgflip.com/30b1gx.jpg"
    }
    
    # --- STEP 1: Routing & Search Query Generation ---
    router_prompt = f"""
    Analyze this situation: "{situation}"
    Target Language: {language}
    
    Generate two search queries:
    1. A text query to find funny tweets/jokes about this topic in English.
    2. An image search query to find a matching blank meme template. 
       - IF Tanglish: Focus on Kollywood actors (e.g., "Vadivelu funny reaction blank", "Santhanam meme template").
       
    Output ONLY valid JSON: {{"text_query": "...", "image_query": "..."}}
    """
    
    try:
        # Ask Groq for search strategy
        res1 = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": router_prompt}],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        queries = json.loads(res1.choices[0].message.content)
        
        # --- STEP 2: The Dangerous Web Scrape (with Try/Except) ---
        internet_humor = ""
        try:
            # Try to get jokes from DDG
            text_results = ddgs.text(queries["text_query"], max_results=2)
            internet_humor = " ".join([r['body'] for r in text_results]) if text_results else ""
        except Exception:
            internet_humor = "Rely on your own internal knowledge." # Fallback if text search 403s
            
        image_url = ""
        used_fallback = False
        try:
            # Try to get image from DDG
            image_results = ddgs.images(queries["image_query"], max_results=1)
            image_url = image_results[0]['image'] if image_results else ""
            if not image_url: raise ValueError("No image found")
        except Exception:
            # 🚨 403 RATE LIMIT CAUGHT: Fallback to Emergency Vault
            used_fallback = True
            # For Tanglish, pick a random Kollywood legend. Otherwise, fallback to a standard.
            if language == "Tanglish":
                image_url = random.choice(list(tamil_vault.values())[:-1]) # Exclude Drake
            else:
                image_url = tamil_vault["Drake (Fallback)"]

        # --- STEP 3: Final Synthesis ---
        tanglish_rules = """
        CRITICAL TANGLISH RULES: 
        - Use strict local Chennai/Madurai slang: Vro, pangu, maapu, sethaya, asingapattan, murugesa.
        - NEVER use formal Tamil. Make it sound like a WhatsApp status.
        - Example 1: "Code compile aagudha nu paaru... Illa azhuthuruven."
        - Example 2: "Naan engineer aaven nu nenacha... IPdi aagitten."
        """ if language == "Tanglish" else "Be unhinged and viral."

        writer_prompt = f"""
        You are an elite meme creator.
        Situation: "{situation}"
        Internet Inspiration: "{internet_humor}"
        
        {tanglish_rules}
        
        Write a hyper-relevant, savage caption (under 12 words) for the situation using the exact requested language ({language}).
        Output ONLY valid JSON: {{"caption": "..."}}
        """
        
        res2 = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": writer_prompt}],
            response_format={"type": "json_object"},
            temperature=0.8
        )
        final_meme = json.loads(res2.choices[0].message.content)
        
        return {
            "caption": final_meme["caption"],
            "image_url": image_url,
            "search_queries": queries,
            "used_fallback": used_fallback
        }
        
    except Exception as e:
        st.error(f"Agent Pipeline Critical Error: {e}")
        return None
def generate_roast_from_vision(image_bytes, language):
    """Sends image to Groq Vision model to get roasted."""
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    tanglish_rules = """
    Use strict local Chennai/Madurai slang: Vro, pangu, maapu.
    NEVER use formal Tamil. Make it sound like a WhatsApp status.
    """ if language == "Tanglish" else "Be unhinged and viral."
    
    prompt = f"""
    Look at this image. You are a savage, Gen-Z meme creator.
    Roast the contents of this image in a short, punchy meme caption (under 12 words).
    Language/Style: {language}.
    {tanglish_rules}
    Output ONLY the raw caption text, no quotes, no JSON, just the roast text.
    """
      
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct", # <--- UPDATED MODEL HERE
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
            temperature=0.8,
            max_tokens=40
        )
        return response.choices[0].message.content.strip(' "')
    except Exception as e:
        st.error(f"Vision AI Failed: {e}")
        return None
def get_best_text_position(img: Image.Image) -> str:
    """Uses OpenCV to detect faces and returns 'top' or 'bottom' for safe text placement."""
    try:
        # 1. Convert PIL image to OpenCV format
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        # 2. Load OpenCV's built-in AI for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # 3. Scan the image for faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # If no face is found, default to the classic bottom text
        if len(faces) == 0:
            return "bottom"

        # 4. Find the largest face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        
        # 5. Calculate center
        face_center_y = y + (h / 2)
        img_h = img.size[1]

        # 6. If face is in bottom half, put text on top
        if face_center_y > (img_h / 2):
            return "top"
        else:
            return "bottom"
            
    except Exception as e:
        print(f"Face detection failed: {e}")
        return "bottom"

# --- 8. MAIN UI ---

main_col1, main_col2, main_col3 = st.columns([1, 2, 1])
with main_col2:
    st.markdown("<h3 style='text-align: center;'>🎙️ Conjure Comedy</h3>", unsafe_allow_html=True)

    tab_classic, tab_vision = st.tabs(["📝 Classic Meme Generator", "📸 Roast My Face (Vision AI)"])

  # TAB 1: CLASSIC
    with tab_classic:
        with st.container(border=True):
            st.markdown("<p style='color: #888; font-size: 14px; font-weight: bold;'>1. Record situation (Optional)</p>", unsafe_allow_html=True)
            audio_value = st.audio_input("Voice Input", label_visibility="collapsed")

            if audio_value:
                if st.button("⚡ Transcribe", key="transcribe_btn", use_container_width=True):
                    with st.spinner("Decoding your voice..."):
                        try:
                            transcription = client.audio.transcriptions.create(
                                file=("audio.wav", audio_value.read()),
                                model="whisper-large-v3",
                                response_format="text",
                            )
                            st.session_state.draft_text = transcription
                            st.rerun()
                        except Exception as e:
                            st.error(f"Transcription Error: {e}")

            st.markdown("<p style='color: #888; font-size: 14px; font-weight: bold; margin-top: 15px;'>2. Type or Edit</p>", unsafe_allow_html=True)
            prompt_input = st.text_area(
                "Input",
                value=st.session_state.draft_text,
                placeholder="E.g., When you get a 9.53 GPA but your code still won't compile...",
                height=80,
                label_visibility="collapsed",
            )
            if prompt_input != st.session_state.draft_text:
                st.session_state.draft_text = prompt_input

            st.markdown("<br>", unsafe_allow_html=True)
            generate_btn = st.button("🧠 Auto-Generate Smart Meme", key="gen_smart_btn", use_container_width=True)

       # THE NEW SMART LOGIC TRIGGER
        if generate_btn and prompt_input:
            with st.spinner("Agent is searching the web for jokes and images..."):
                
                # Call our new advanced pipeline
                result = internet_meme_agent(prompt_input, meme_language)
                
                if result and "caption" in result and "image_url" in result:
                    caption = result["caption"]
                    img_url = result["image_url"]
                    
                    #st.success(f"🎯 Web Search Used: {result['search_queries']['image_query']}")
                    
                # THE NEW SMART LOGIC TRIGGER
        if generate_btn and prompt_input:
            with st.spinner("Agent is searching the web for jokes and images..."):
                
                # Call our new advanced pipeline
                result = internet_meme_agent(prompt_input, meme_language)
                
                if result and "caption" in result and "image_url" in result:
                    caption = result["caption"]
                    img_url = result["image_url"]
                    
                    st.success(f"🎯 Web Search Used: {result['search_queries']['image_query']}")
                    
                    try:
                        # 1. Fetch the image from the web
                        headers = {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                            'Accept': 'image/jpeg,image/png,image/webp,image/*;q=0.8'
                        }
                        img_res = requests.get(img_url, headers=headers, timeout=10)
                        img_res.raise_for_status() 
                        img = Image.open(io.BytesIO(img_res.content)).convert("RGB")
                         
                    except Exception:
                        st.toast("⚠️ Template blocked! Forcing AI to draw it instead...", icon="✨")
                        try:
                            # 🚨 THE NEW AI IMAGE FALLBACK
                            safe_prompt = urllib.parse.quote(f"A funny reaction meme background about: {prompt_input}, highly detailed, no text, no words")
                            ai_url = f"https://image.pollinations.ai/prompt/{safe_prompt}?width=800&height=800&nologo=true"
                            
                            ai_img_res = requests.get(ai_url, timeout=15)
                            ai_img_res.raise_for_status()
                            img = Image.open(io.BytesIO(ai_img_res.content)).convert("RGB")
                            
                        except Exception:
                            # The absolute last resort if both the internet AND the AI fail
                            st.toast("⚠️ AI generation also failed. Using a blank canvas.", icon="🚧")
                            img = Image.new('RGB', (800, 800), color=(20, 20, 25))
                    
                    # 2. Burn the text onto whatever image we successfully got
                    try:
                        final_meme = burn_meme_text(img, caption, meme_text_color)
                        
                        # Display and Save
                        st.image(final_meme, use_container_width=True)
                        st.session_state.meme_history.append(
                            {"image": final_meme, "caption": caption}
                        )
                    except Exception as e:
                        st.error(f"Image Pipeline Error: {e}")
    # TAB 2: VISION
    with tab_vision:
        with st.container(border=True):
            upload_pic = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
            camera_pic = st.camera_input("Take a Selfie", label_visibility="collapsed")

            user_img_source = camera_pic if camera_pic else upload_pic

            if user_img_source:
                # 🔥 THE FIX: Use getvalue() so the image survives the button click!
                img_bytes = user_img_source.getvalue() 
                user_pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                st.image(user_pil_img, caption="Target Locked 🎯", use_container_width=True)

                if st.button("🔥 Roast This Image", key="roast_btn", use_container_width=True):
                    with st.spinner("Scanning for emotional damage..."):
                        
                        roast_caption = generate_roast_from_vision(img_bytes, meme_language)
                        
                        if roast_caption:
                            st.info(f"💬 AI Caption: '{roast_caption}'")
                            
                            # 1. Ask OpenCV where the face is
                            safe_zone = get_best_text_position(user_pil_img)
                            
                            if safe_zone == "top":
                                st.toast("🎯 Face detected at the bottom! Moving text to the top.", icon="⬆️")
                            
                            # 2. Tell the text burner where to put it
                            final_roast = burn_meme_text(user_pil_img.copy(), roast_caption, meme_text_color, position=safe_zone)
                            
                            st.image(final_roast, use_container_width=True)
                            
                            # Save to vault
                            st.session_state.meme_history.append({"image": final_roast, "caption": roast_caption})
                            st.success("Roast complete!")

# --- 9. MEME VAULT ---
st.markdown("---")
col_title, col_clear = st.columns([4, 1], vertical_alignment="bottom")
with col_title:
    st.markdown("### 📜 Your Meme Vault")
with col_clear:
    if st.session_state.meme_history:
        if st.button("🗑️ Clear Vault", use_container_width=True):
            st.session_state.meme_history = []
            st.rerun()

if not st.session_state.meme_history:
    st.info("Your vault is empty! Generate memes and they'll appear here.")
else:
    history_cols = st.columns(3)
    for index, past_meme in enumerate(reversed(st.session_state.meme_history)):
        original_index = len(st.session_state.meme_history) - 1 - index
        with history_cols[index % 3]:
            with st.container(border=True):
                st.image(past_meme["image"], use_container_width=True)
                st.markdown(f"<div style='font-style: italic; font-size: 14px; margin-top: 10px;'>&quot;{past_meme['caption']}&quot;</div>", unsafe_allow_html=True)
                
                buf = io.BytesIO()
                past_meme["image"].save(buf, format="PNG")
                st.download_button(
                    label="⬇️ Download",
                    data=buf.getvalue(),
                    file_name=f"memegenz_{original_index}.png",
                    mime="image/png",
                    use_container_width=True,
                    key=f"dl_{original_index}",
                )
                
                viral_message = f"Bro look at this meme I made! 😂\n\n\"{past_meme['caption']}\""
                encoded_message = urllib.parse.quote(viral_message)
                st.markdown(f'<a href="https://wa.me/?text={encoded_message}" target="_blank" style="display: block; text-align: center; background-color: #25D366; color: white; padding: 6px; border-radius: 8px; text-decoration: none; font-weight: bold; margin-top: 5px;">🟢 Share on WhatsApp</a>', unsafe_allow_html=True)
                
                if st.button("❌ Remove", key=f"del_{original_index}", use_container_width=True):
                    st.session_state.meme_history.pop(original_index)
                    st.rerun()