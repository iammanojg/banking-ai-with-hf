# llm_local.py — FINAL, BULLETPROOF VERSION (Chat Completions API)

import os
import streamlit as st
from huggingface_hub import InferenceClient

# ------------------- STATIC FALLBACKS -------------------
STATIC_FALLBACK = {
    'Cash': "Tip: Consider using a Credit Card for larger purchases to earn rewards and build credit.",
    'Debit Card': "Insight: Debit is practical for daily spending. A Credit Card can offer cashback, rewards, and better buyer protections when used responsibly.",
    'Credit Card': "Excellent Choice! You're optimizing for rewards and protection. Consider paying in full each month to avoid interest.",
    'Digital Wallet': "Smart move! Link your Digital Wallet to a rewards Credit Card to earn points while keeping convenience."
}

# ------------------- READ TOKEN CORRECTLY -------------------
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "gpt2")

# ------------------- PROMPT (Now as Chat Message) -------------------
def _prompt_for_customer(summary: dict) -> str:
    summary_str = ", ".join(f"{k}: {v}" for k, v in summary.items())
    return (
        "You are a friendly, concise financial advisor. Write a short, natural recommendation (40–100 words) "
        "encouraging this customer to consider a rewards Credit Card. Highlight 1–2 benefits based on their behavior. "
        "End with a clear call-to-action. Never mention personal data.\n\n"
        f"Customer behavior: {summary_str}\n\n"
        "Recommendation:"
    )

# ------------------- HF CHAT COMPLETIONS CALL (NEW OFFICIAL METHOD) -------------------
def generate_with_hf_inference(customer_summary: dict, max_tokens: int = 150, temperature: float = 0.7) -> str:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN not set in environment.")

    client = InferenceClient(token=HF_TOKEN)
    prompt = _prompt_for_customer(customer_summary)

    # FIXED: Use chat.completions.create (the unified 2025 endpoint)
    completion = client.chat.completions.create(
        model=HF_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature
    )

    return completion.choices[0].message.content.strip()

# ------------------- SAFE PUBLIC FUNCTION -------------------
@st.cache_data(show_spinner=False, ttl=60*60)
def safe_generate_tip(customer_summary: dict, fallback_label: str) -> str:
    if HF_TOKEN:
        try:
            generated = generate_with_hf_inference(customer_summary)
            if generated and len(generated) > 15:
                return generated
        except Exception as e:
            st.warning(f"Hugging Face generation failed → using fallback ({e})")

    return STATIC_FALLBACK.get(fallback_label, "Consider a rewards Credit Card for extra benefits!")
