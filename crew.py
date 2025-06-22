
from crewai import Agent, Task, Crew, Process
from llm_wrapper import HuggingFaceLLM

# Shared LLM instance
llm = HuggingFaceLLM()

# Define Venue Finder Agent
venue_finder_agent = Agent(
    role="Venue Finder",
    goal="Find the best 5 venues for a conference in Las Vegas",
    backstory="You are an expert event planner with great knowledge of Las Vegas venues.",
    allow_delegation=False,
    verbose=True,
    llm=llm
)

# Define Venue Quality Checker Agent
venue_qa_agent = Agent(
    role="Venue Quality Assurance",
    goal="Review venues for quality and recommend the top choices",
    backstory="You specialize in validating venues for large-scale events, ensuring all requirements are met.",
    allow_delegation=False,
    verbose=True,
    llm=llm
)

# Define the first task (venue search)
venue_search_task = Task(
    description=(
        "Find 5 top venues for the upcoming '{conference_name}' in Las Vegas.\n"
        "Requirements: {requirements}.\n"
        "Provide name, capacity, location, amenities, price, and availability."
    ),
    expected_output="List of 5 venues in detailed bullet format",
    agent=venue_finder_agent
)

# Define the second task (venue QA review)
venue_qa_task = Task(
    description=(
        "Evaluate the list of venues found for '{conference_name}'.\n"
        "Requirements: {requirements}.\n"
        "Assess quality, identify issues, and recommend the top 2 venues with reasoning."
    ),
    expected_output="QA report with analysis and final recommendation",
    agent=venue_qa_agent
)

# Assemble the crew
crew = Crew(
    agents=[venue_finder_agent, venue_qa_agent],
    tasks=[venue_search_task, venue_qa_task],
    process=Process.sequential,
    verbose=True
)

def run_crew(conference_name, requirements):
    inputs = {
        "conference_name": conference_name,
        "requirements": requirements
    }
    result = crew.kickoff(inputs=inputs)
    return result
