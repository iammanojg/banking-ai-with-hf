# llm_local.py — FIXED WITH InferenceClient (Official HF 2025 Method)

import os
import streamlit as st
import requests
from huggingface_hub import InferenceClient  # ← NEW: Official client

# ------------------- STATIC FALLBACKS -------------------
STATIC_FALLBACK = {
    'Cash': "Tip: Consider using a Credit Card for larger purchases to earn rewards and build credit.",
    'Debit Card': "Insight: Debit is practical for daily spending. A Credit Card can offer cashback, rewards, and better buyer protections when used responsibly.",
    'Credit Card': "Excellent Choice! You're optimizing for rewards and protection. Consider paying in full each month to avoid interest.",
    'Digital Wallet': "Smart move! Link your Digital Wallet to a rewards Credit Card to earn points while keeping convenience."
}

# ------------------- READ TOKEN CORRECTLY -------------------
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "HuggingFaceH4/zephyr-7b-beta")

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

# ------------------- HF INFERENCE CALL (NEW OFFICIAL METHOD) -------------------
def generate_with_hf_inference(customer_summary: dict, max_tokens: int = 150, temperature: float = 0.7) -> str:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN not set in environment.")

    client = InferenceClient(token=HF_TOKEN)  # Handles routing automatically
    prompt = _prompt_for_customer(customer_summary)

    result = client.text_generation(
        prompt,
        model=HF_MODEL,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True,
        return_full_text=False
    )

    return result.strip()

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
