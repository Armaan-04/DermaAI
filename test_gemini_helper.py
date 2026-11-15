
from utils.gemini_helper import GeminiExplainer

explainer = GeminiExplainer()

result = explainer.explain_condition("Eczema")

print("\nGEMINI EXPLANATION TEST SUCCESSFUL\n")
print(result)
