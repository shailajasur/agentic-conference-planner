
from llm_wrapper import HuggingFaceLLM

class VenueQualityCheckerAgent:
    def __init__(self):
        self.llm = HuggingFaceLLM()

    def run(self, venue_summary):
        prompt = f"""
You are a venue quality assurance specialist. Review the following list of venue options for an upcoming conference and provide:

- Overall strengths
- Any concerns or gaps
- Recommend the best 1 or 2 venues with reasons

Venue List:
{venue_summary}

Your output should be in a structured and clear review format.
"""
        return self.llm.generate(prompt)
