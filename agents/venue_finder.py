
from llm_wrapper import HuggingFaceLLM

class VenueFinderAgent:
    def __init__(self):
        self.llm = HuggingFaceLLM()

    def run(self, conference_name, requirements):
        prompt = f"""
You are an expert event planner. Find 5 excellent venues in Las Vegas, USA for the following conference:

Conference Name: {conference_name}
Requirements: {requirements}

For each venue, include:
- Name
- Capacity
- Location
- Key Amenities
- Estimated Pricing
- Availability (assumed for a date in the next 6 months)

Output in a readable bullet list format.
"""
        return self.llm.generate(prompt)
