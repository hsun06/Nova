import streamlit as st
from openai import OpenAI

PRESETS = {
    "Nova (Default)": """You are Nova: sharp, pragmatic, structured.
Rules:
- Be concise but complete.
- Prefer bullet points, checklists, and templates.
- Ask up to 3 targeted questions if key info is missing.
- If uncertain, say what you’re unsure about.
- End with "Next steps" when user asks for a plan.""",
    "Nova — MBA / Finance": """You are Nova, an MBA/finance copilot.
Use assumptions → drivers → logic → conclusion. Include risks.""",
    "Nova — Lawyer Mode": """You are Nova, a careful legal drafting assistant (not a lawyer).
Separate Facts / Issues / Options / Risks / Draft Language.""",
    "Nova — Engineering Mode": """You are Nova, an engineering copilot.
Be precise, step-by-step, verification-focused.""",
}

st.set_page_config(page_title="Nova", page_icon="✨", layout="wide")
st.title("✨ Nova")
st.caption("Public Nova (Streamlit Cloud) + hosted model (OpenAI).")

# --- Secrets check
if "OPENAI_API_KEY" not in st.secrets:
    st.error("Missing OPENAI_API_KEY in Streamlit Secrets. Add it in Streamlit Cloud → App → Settings → Secrets.")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- Sidebar
with st.sidebar:
    st.header("Settings")
    model = st.text_input("Model", value="gpt-4.1-mini")
    preset_name = st.selectbox("Nova preset", list(PRESETS.keys()), index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)

    st.divider()
    if st.button("New session", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

# --- State
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# --- Render history
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- Chat
user_text = st.chat_input("Message Nova…")
if user_text:
    st.session_state["messages"].append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    system_prompt = PRESETS[preset_name].strip()
    messages = [{"role": "system", "content": system_prompt}] + st.session_state["messages"]

    with st.chat_message("assistant"):
        try:
            resp = client.responses.create(
                model=model,
                input=messages,
                temperature=temperature,
            )
            assistant_text = resp.output_text
        except Exception as e:
            assistant_text = f"⚠️ OpenAI API error: {e}"

        st.markdown(assistant_text)

    st.session_state["messages"].append({"role": "assistant", "content": assistant_text})
