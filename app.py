
import streamlit as st
from crew import run_crew

st.title("🧠 Agentic Conference Venue Planner")

st.markdown("This app uses a multi-agent AI system to find and evaluate top venues for your event.")

conference_name = st.text_input("Conference Name", "AI Innovations Summit")
requirements = st.text_area("Requirements", "Capacity for 5000, central location, modern amenities, budget up to $50,000")

if st.button("Run Agentic Workflow"):
    with st.spinner("Running CrewAI agents..."):
        result = run_crew(conference_name, requirements)
        st.subheader("🏁 Final Output")
        st.markdown(result)
