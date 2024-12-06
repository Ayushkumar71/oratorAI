


import streamlit as st
from pages.home import home  
from pages.results import results  
from pages.help import help

def main():
    # Initialize session state for page navigation if not already set
    if "page" not in st.session_state:
        st.session_state["page"] = "home"  # Default page (home)

    # Sidebar for page navigation
    st.sidebar.title("Navigation")
    page_selection = st.sidebar.radio("Go to page", ("Home", "Results", "Help"))

    # Update the session state page selection based on sidebar input
    if page_selection == "Home":
        st.session_state["page"] = "home"
    elif page_selection == "Results":
        st.session_state["page"] = "results"
    elif page_selection == "Help":
        st.session_state["page"] = "help"

    # Handle page navigation based on session state
    if st.session_state["page"] == "home":
        home()  # Function for the home page where recording happens
    elif st.session_state["page"] == "results":
        results()  # Function to show results analysis from results.py
    elif st.session_state["page"] == "help":
        help()  # Function to show tips for improving user's presentation 

if __name__ == "__main__":
    main()





