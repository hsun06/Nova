import json
import time
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st


# ----------------------------
# Config
# ----------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1:8b"  # change to llama3.2:3b if you want faster/lighter
SAVE_DIR = Path.home() / "nova_ui" / "sessions"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Nova Layer (System Prompts)
# ----------------------------
PRESETS = {
    "Nova (Default)": """You are Nova: a sharp, pragmatic, structured assistant.
Rules:
- Be concise but complete.
- Prefer bullet points, checklists, and templates.
- If key info is missing, ask up to 3 targeted questions.
- If uncertain, say what you’re unsure about and what would confirm it.
- End with "Next steps" when the user is asking for a plan.""",

    "Nova — MBA / Finance": """You are Nova, acting as an MBA/finance copilot.
Rules:
- Use structured thinking: assumptions → drivers → math/logic → conclusion.
- Provide models/frameworks (DCF, comps, LBO, WACC) when relevant.
- Include a short “Risks / Watch-outs” section for decisions.
- Be concise; use bullets and tables (plain text) where helpful.""",

    "Nova — Lawyer Mode": """You are Nova, acting as a careful legal drafting assistant (not a lawyer).
Rules:
- Ask clarifying questions when jurisdiction/facts matter.
- Separate: Facts / Issues / Options / Risks / Suggested Draft Language.
- Avoid overconfident statements; flag uncertainty.
- For drafts, use clean headings and formal tone.""",

    "Nova — Engineering Mode": """You are Nova, an engineering copilot.
Rules:
- Be technical, precise, and action-oriented.
- When debugging, ask for minimal reproducible details.
- Provide step-by-step procedures and verification checks.
- Use numbered steps and short code snippets when helpful.""",
}


# ----------------------------
# Helpers
# ----------------------------
def ollama_tags():
    """List models available in Ollama."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
        data = r.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def ollama_chat(model: str, messages: list[dict], temperature: float = 0.3):
    """
    Stream a chat completion from Ollama.
    Uses /api/chat streaming endpoint.
    """
    payload = {
        "model": model,
        "messages": messages,
        "options": {"temperature": temperature},
        "stream": True,
    }

    with requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, stream=True, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            chunk = json.loads(line.decode("utf-8"))
            if "message" in chunk and "content" in chunk["message"]:
                yield chunk["message"]["content"]
            if chunk.get("done"):
                break


def export_markdown(title: str, messages: list[dict]) -> str:
    out = [f"# {title}", ""]
    for m in messages:
        role = m["role"].capitalize()
        content = m["content"].strip()
        out.append(f"## {role}\n\n{content}\n")
    return "\n".join(out).strip() + "\n"


def autosave_session(session_title: str, messages: list[dict]):
    safe = "".join([c if c.isalnum() or c in (" ", "-", "_") else "_" for c in session_title]).strip()
    if not safe:
        safe = "Nova Session"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SAVE_DIR / f"{safe}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "title": session_title,
                "saved_at": datetime.now().isoformat(),
                "messages": messages,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return path


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Nova", page_icon="✨", layout="wide")

st.title("✨ Nova")
st.caption("Local, private assistant running on your machine via Ollama.")

# Sidebar
with st.sidebar:
    st.header("Settings")

    available_models = ollama_tags()
    if available_models:
        model = st.selectbox("Model", available_models, index=available_models.index(DEFAULT_MODEL) if DEFAULT_MODEL in available_models else 0)
    else:
        model = st.text_input("Model (Ollama name)", value=DEFAULT_MODEL)
        st.warning("Could not fetch Ollama models. Make sure Ollama is running and reachable at localhost:11434.")

    preset_name = st.selectbox("Nova preset", list(PRESETS.keys()), index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)

    st.divider()

    st.subheader("Session")
    session_title = st.text_input("Session title", value=st.session_state.get("session_title", "Nova Session"))
    st.session_state["session_title"] = session_title

    colA, colB = st.columns(2)
    with colA:
        if st.button("New session", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["session_started_at"] = time.time()
            st.rerun()
    with colB:
        if st.button("Clear chat", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

    autosave = st.checkbox("Auto-save on export", value=True)

    st.divider()
    st.subheader("Export")
    if st.session_state.get("messages"):
        md = export_markdown(session_title, st.session_state["messages"])
        st.download_button(
            label="Download Markdown",
            data=md.encode("utf-8"),
            file_name=f"{session_title.replace(' ', '_')}.md",
            mime="text/markdown",
            use_container_width=True,
        )
        if autosave and st.button("Save JSON locally", use_container_width=True):
            path = autosave_session(session_title, st.session_state["messages"])
            st.success(f"Saved to: {path}")
    else:
        st.info("Start chatting to enable export.")

# Init state
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "session_started_at" not in st.session_state:
    st.session_state["session_started_at"] = time.time()

# Render existing messages
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input
user_text = st.chat_input("Message Nova…")

if user_text:
    # Build prompt with system preset + history
    system_prompt = PRESETS[preset_name].strip()

    # Store user msg
    st.session_state["messages"].append({"role": "user", "content": user_text})

    # Display user msg
    with st.chat_message("user"):
        st.markdown(user_text)

    # Assistant response streaming
    with st.chat_message("assistant"):
        placeholder = st.empty()
        assistant_text = ""

        # Compose messages for Ollama: include system first
        send_messages = [{"role": "system", "content": system_prompt}] + st.session_state["messages"]

        try:
            for token in ollama_chat(model=model, messages=send_messages, temperature=temperature):
                assistant_text += token
                placeholder.markdown(assistant_text)
        except requests.exceptions.RequestException as e:
            assistant_text = f"⚠️ Nova couldn’t reach Ollama at {OLLAMA_BASE_URL}.\n\nError: {e}\n\nMake sure Ollama is running and the model exists."
            placeholder.markdown(assistant_text)

    st.session_state["messages"].append({"role": "assistant", "content": assistant_text})

