import streamlit as st
import os
import json
import time
# Dynamic import for optional genai dependency (avoids static import errors in editors)
import importlib
try:
    genai = importlib.import_module("google.genai")
except Exception:
    try:
        genai = importlib.import_module("genai")
    except Exception:
        genai = None

# --- PAGE SETUP ---
st.set_page_config(page_title="LEX Model Trainer", page_icon="⚖️", layout="wide")
st.title("⚖️ LEX Custom Model Training Hub")
st.caption("A standalone Streamlit sandbox to parse custom FAQ data or run live generative AI modules.")

# --- SIDEBAR UI ---
st.sidebar.header("⚙️ Configuration Controls")

# 1. Select Model Option Dropdown
model_choice = st.sidebar.selectbox(
    "Select Intelligence Model",
    ["Lex-Local-Semantic-Matcher", "gemini-2.5-flash"]
)

# 2. Upload FAQs File Option Component
uploaded_file = st.sidebar.file_uploader("Upload FAQs Knowledge Base File", type=["json", "txt"])

# --- SESSION STATE INITIALIZATION ---
if "is_trained" not in st.session_state:
    st.session_state.is_trained = False
if "faq_database" not in st.session_state:
    st.session_state.faq_database = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hello! I am LEX. Please upload an FAQ sheet on the left panel and click train to start testing your model."}
    ]

# --- LIGHTWEIGHT PARSING & TRAINING PIPELINE ---
if uploaded_file is not None and not st.session_state.is_trained:
    if st.sidebar.button("🚀 Execute Training Pipeline"):
        with st.sidebar.status("Parsing file contents and structuring keys...", expanded=True) as status:
            try:
                raw_content = uploaded_file.read().decode("utf-8")
                
                # Check if file format is structured JSON or plain text strings
                if uploaded_file.name.endswith(".json"):
                    st.session_state.faq_database = json.loads(raw_content)
                else:
                    lines = [line.strip() for line in raw_content.split("\n") if line.strip()]
                    st.session_state.faq_database = lines
                
                time.sleep(1.5) # Simulating training/embedding computation delay
                
                status.update(label="Training complete!", state="complete", expanded=False)
                st.session_state.is_trained = True
                st.sidebar.success("✅ Model Core Trained Successfully!")
            except Exception as e:
                status.update(label="Training compilation failed.", state="error")
                st.sidebar.error(f"Error: {str(e)}")

# --- MAIN SCREEN CHATBOT SYSTEM ---
st.subheader("💬 Live Model Inference Testing Screen")

# Display historical message exchanges
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Handle live inputs via the terminal style input component
if user_query := st.chat_input("Ask your trained model a question..."):
    
    # Render user query bubble instantly
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # Process and render response bubble
    with st.chat_message("assistant"):
        with st.spinner("Lex engine computing weights..."):
            execution_response = ""
            query_normalized = user_query.lower()

            # Mode A: Context-Aware Matching over uploaded sheet arrays
            if model_choice == "Lex-Local-Semantic-Matcher":
                if st.session_state.is_trained:
                    match_found = False
                    for record in st.session_state.faq_database:
                        if isinstance(record, dict):
                            # Flexibly fall back across common dataset JSON keys
                            q_key = record.get("question", "").lower() or record.get("instruction", "").lower()
                            a_key = record.get("answer", "") or record.get("output", "")
                            
                            if q_key and q_key in query_normalized:
                                execution_response = f"**[Verified Local Dataset Match]**\n\n{a_key}"
                                match_found = True
                                break
                    if not match_found:
                        execution_response = "Context parsed, but no matching question-answer signature found in the uploaded FAQ file."
                else:
                    execution_response = "❌ System is untrained. Please upload an FAQ document and click 'Execute Training Pipeline' in the sidebar."

            # Mode B: Fallback to Live Cloud Gemini AI Complete Module
            elif model_choice == "gemini-2.5-flash":
                try:
                    runtime_key = os.getenv("GEMINI_API_KEY")
                    if runtime_key:
                        client = genai.Client(api_key=runtime_key)
                        completion = client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=user_query
                        )
                        execution_response = completion.text
                    else:
                        execution_response = "⚠️ Cloud check failed. GEMINI_API_KEY missing from environment variables configuration."
                except Exception as api_err:
                    execution_response = f"API Structural Failure: {str(api_err)}"

            st.write(execution_response)
            st.session_state.chat_history.append({"role": "assistant", "content": execution_response})