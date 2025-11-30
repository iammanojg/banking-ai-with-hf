# llm_local.py — FINAL WORKING VERSION (Streamlit Cloud + Local)

import os
import streamlit as st
import requests

# ------------------- STATIC FALLBACKS -------------------
STATIC_FALLBACK = {
    'Cash': "Tip: Consider using a Credit Card for larger purchases to earn rewards and build credit.",
    'Debit Card': "Insight: Debit is practical for daily spending. A Credit Card can offer cashback, rewards, and better buyer protections when used responsibly.",
    'Credit Card': "Excellent Choice! You're optimizing for rewards and protection. Consider paying in full each month to avoid interest.",
    'Digital Wallet': "Smart move! Link your Digital Wallet to a rewards Credit Card to earn points while keeping convenience."
}

# ------------------- READ TOKEN CORRECTLY -------------------
# Streamlit Cloud uses HF_TOKEN → we also support the LangChain name
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Use a FAST, FREE, always-working model by default
HF_MODEL = os.getenv("HF_MODEL", "HuggingFaceH4/zephyr-7b-beta")  # ← super fast & excellent


# ------------------- PROMPT -------------------
def _prompt_for_customer(summary: dict) -> str:
    summary_str = ", ".join(f"{k}: {v}" for k, v in summary.items())
    prompt = (
        "You are a friendly, concise financial advisor. Write a short, natural recommendation (40–100 words) "
        "encouraging this customer to consider a rewards Credit Card. Highlight 1–2 benefits based on their behavior. "
        "End with a clear call-to-action. Never mention personal data.\n\n"
        f"Customer behavior: {summary_str}\n\n"
        "Recommendation:"
    )
    return prompt


# ------------------- HF INFERENCE CALL -------------------
def generate_with_hf_inference(customer_summary: dict, max_tokens: int = 150, temperature: float = 0.7) -> str:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN not set in environment.")

    prompt = _prompt_for_customer(customer_summary)
    api_url = f"https://router.huggingface.co/hf-inference/{HF_MODEL}"

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "return_full_text": False
        }
    }

    response = requests.post(api_url, headers=headers, json=payload, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"HF API error {response.status_code}: {response.text}")

    data = response.json()
    if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
        return data[0]["generated_text"].strip()
    elif isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"].strip()
    else:
        return str(data).strip()


# ------------------- SAFE PUBLIC FUNCTION -------------------
@st.cache_data(show_spinner=False, ttl=60*60)  # cache 1 hour
def safe_generate_tip(customer_summary: dict, fallback_label: str) -> str:
    # Try real Hugging Face generation
    if HF_TOKEN:
        try:
            generated = generate_with_hf_inference(customer_summary)
            if generated and len(generated) > 15:
                return generated
        except Exception as e:
            st.warning(f"Hugging Face generation failed → using fallback ({e})")

    # Fallback to static message
    return STATIC_FALLBACK.get(fallback_label, "Consider a rewards Credit Card for extra benefits!")
