import streamlit as st

def setup_sidebar():
    st.sidebar.title("🔧 Interface Selection", help="Select the interface mode to configure the application settings.")
    return st.sidebar.radio(
        "Choose Mode", 
        ["Admin Interface", "Client Interface"], 
        help="Select 'Admin Interface' for administrative tasks or 'Client Interface' for client-related operations."
    )
