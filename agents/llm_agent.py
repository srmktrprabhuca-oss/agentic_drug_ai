import os
from groq import Groq

# Load the API key from environment variable or Streamlit secrets (for local development)
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    try:
        import streamlit as st
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass

if not api_key:
    raise ValueError("GROQ_API_KEY not found. Please set it in environment variables or .streamlit/secrets.toml.")

client = Groq(api_key=api_key)

def generate_ai_report(data, prediction, confidence):

    risk_text = "HIGH RISK" if prediction == 1 else "LOW RISK"

    prompt = f"""
You are a clinical AI assistant.

Generate a structured medical report EXACTLY in this format:

**Patient Risk Assessment Report**

**1. Patient Information**
- Drugs: {data['drugs']}
- Glucose: {data['glucose']}
- BP: {data['bp']}
- Cholesterol: {data['cholesterol']}
- Kidney: {data['kidney']}
- HbA1c: {data['hba1c']}
- BMI: {data['bmi']}
- HDL: {data['hdl']}
- LDL: {data['ldl']}
- Ulcer: {data['ulcer']}

**2. Risk**
- Risk Level: {risk_text}
- Confidence: {confidence}
- Explanation: Explain clearly why the patient falls into this category

**3. Drug Support Analysis**
- Evaluate how each drug contributes to improving the patient’s condition
- Clearly mention:
  - Which drugs are appropriate for the given parameters
  - Which drugs are weak/less effective for this condition
  - Any mismatch between drugs and patient condition
  - Overall effectiveness of the drug combination
- Provide a final summary:
  - Support Level: (High / Moderate / Low)
  - Reason: Explain why

**4. Organs Affected**
- Mention affected organs based on patient condition

**5. Complications**
- List possible complications

**6. Precautions**
- List precautions (NO prescriptions, only guidance)

**7. Lifestyle Advice**
- Provide lifestyle and behavioral advice

Instructions:
- ALL headings MUST be in **bold**
- ALL content MUST be in bullet points (-)
- Be clinical and detailed
- Do NOT write paragraphs
- For Drug Support Analysis:
  - MUST be detailed and point-by-point
  - MUST include reasoning for EACH drug
- Keep everything structured and readable
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content