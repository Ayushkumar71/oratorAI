import streamlit as st
from pages.home import home  # Import the function to run facecam (for the home page)
from pages.results import results  # Assuming you have this function in results.py

def main():
    # Initialize session state for page navigation if not already set
    if "page" not in st.session_state:
        st.session_state["page"] = "home"  # Default page (home)

    # Sidebar for page navigation
    st.sidebar.title("Navigation")
    page_selection = st.sidebar.radio("Go to page", ("Home", "Results"))

    # Update the session state page selection based on sidebar input
    if page_selection == "Home":
        st.session_state["page"] = "home"
    elif page_selection == "Results":
        st.session_state["page"] = "results"

    # Handle page navigation based on session state
    if st.session_state["page"] == "home":
        home()  # Function for the home page where recording happens
    elif st.session_state["page"] == "results":
        results()  # Function to show results analysis from results.py

if __name__ == "__main__":
    main()