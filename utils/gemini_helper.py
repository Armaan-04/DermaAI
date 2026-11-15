
import streamlit as st
from google import genai


class GeminiExplainer:

    def __init__(self):

        api_key = st.secrets["GEMINI_API_KEY"]

        self.client = genai.Client(api_key=api_key)

    def explain_condition(self, condition):

        prompt = f"""
        You are an educational skin-health information assistant
        for a project called DermaAI.

        The AI image classifier has predicted: {condition}.

        Provide an easy-to-understand educational explanation
        using these sections:

        1. What is this condition?
        2. Common symptoms
        3. Possible causes and triggers
        4. General skincare and care suggestions
        5. When should someone consult a dermatologist?

        Important rules:
        - Use simple, clear language.
        - Do not claim that the user definitely has this condition.
        - Do not prescribe medication or dosages.
        - Mention that the information is educational, not a diagnosis.
        - Do not assume the uploaded image confirms the condition.
        """

        response = self.client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )

        return response.text
