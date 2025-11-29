import os
import streamlit as st

STATIC_FALLBACK = {
    'Cash': "Tip: Consider using a Credit Card for larger purchases to earn rewards and build credit.",
    'Debit Card': "Insight: Debit is practical for daily spending. A Credit Card can offer cashback, rewards, and better buyer protections when used responsibly.",
    'Credit Card': "Excellent Choice! You're optimizing for rewards and protection. Consider paying in full each month to avoid interest.",
    'Digital Wallet': "Smart move! Link your Digital Wallet to a rewards Credit Card to earn points while keeping convenience."
}

# read env vars
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "gpt2")

def _prompt_for_customer(summary: dict):
    # build a compact non-PII prompt
    summary_str = ", ".join(f"{k}: {v}" for k, v in summary.items())
    prompt = (
        "You are a friendly, concise financial advisor writing a short message (40-120 words) "
        "to encourage this customer to consider a rewards Credit Card. Mention 1-2 benefits relevant to the customer's behavior and finish with a single call-to-action. "
        "Do NOT include any sensitive personal information.\n\n"
        f"Customer summary: {summary_str}\n\n"
        "Output:"
    )
    return prompt

def generate_with_hf_inference(customer_summary: dict, max_tokens: int = 150, temperature: float = 0.7):
    """
    Call Hugging Face Inference API. Requires HF_TOKEN in environment.
    """
    import requests
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN not set in environment.")
    prompt = _prompt_for_customer(customer_summary)
    api_url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens, "temperature": temperature}}
    r = requests.post(api_url, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Hugging Face inference error {r.status_code}: {r.text}")
    data = r.json()
    # extract generated text from common formats
    text = ""
    if isinstance(data, list) and len(data) > 0:
        # common: [{'generated_text': '...'}]
        if isinstance(data[0], dict) and "generated_text" in data[0]:
            text = data[0]["generated_text"]
        else:
            # fallback: try to stringify
            text = str(data[0])
    elif isinstance(data, dict) and "generated_text" in data:
        text = data["generated_text"]
    else:
        text = str(data)
    return text.strip()

@st.cache_data(show_spinner=False, ttl=60*60)
def safe_generate_tip(customer_summary: dict, fallback_label: str):
    # 1) Try HF inference
    try:
        text = generate_with_hf_inference(customer_summary)
        if text and len(text) > 10:
            return text
    except Exception as e:
        # don't crash the app — show a small warning and fall back
        st.warning(f"Hugging Face generation failed (falling back): {e}")
    # 2) fallback static message
    return STATIC_FALLBACK.get(fallback_label, "Consider a rewards Credit Card for benefits and protection.")
