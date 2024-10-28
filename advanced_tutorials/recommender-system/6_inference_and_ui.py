import streamlit as st
import os
from functions.utils import get_deployments
from functions.recommenders import customer_recommendations, llm_recommendations
from functions.interaction_tracker import get_tracker
from functions.feature_group_updater import get_fg_updater
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CUSTOMER_IDS = [
    '641e6f3ef3a2d537140aaa0a06055ae328a0dddf2c2c0dd6e60eb0563c7cbba0',
    '1fdadbb8aa9910222d9bc1e1bd6fb1bd9a02a108cb0e899b640780f32d8f7d83',
    '7b0621c12c65570bdc4eadd3fca73f081e2da5769f0d31585ac301cea58af53f',
    '675cd49509ef9692d793af738c08d9bce0856036b9e988cba4e26422944314d6',
    '895576481a1095ad66ab3279483f4323724e9d53d9f089b16f289a3f660c1101',
]

def initialize_page():
    """Initialize Streamlit page configuration"""
    st.set_page_config(layout="wide", initial_sidebar_state='expanded')
    st.title('👒 Fashion Items Recommender')
    st.sidebar.title("⚙️ Configuration")

def initialize_services():
    """Initialize tracker, updater, and deployments"""
    tracker = get_tracker()
    fg_updater = get_fg_updater(batch_size=50)
    
    logger.info("Initializing deployments...")
    with st.sidebar:
        with st.spinner("🚀 Starting Deployments..."):
            articles_fv, ranking_deployment, query_model_deployment = get_deployments()
        st.success('✅ Deployments Ready')
        
        # Stop deployments button
        if st.button("⏹️ Stop Deployments", key='stop_deployments_button', type="secondary"):
            ranking_deployment.stop()
            query_model_deployment.stop()
            st.success("Deployments stopped successfully!")
    
    return tracker, fg_updater, articles_fv, ranking_deployment, query_model_deployment

def show_interaction_dashboard(tracker, fg_updater, page_selection):
    """Display interaction data and controls"""
    with st.sidebar.expander("📊 Interaction Dashboard", expanded=True):
        # OpenAI API Key input moved here for better visibility
        if page_selection == "LLM Recommendations":
            api_key = st.text_input("🔑 OpenAI API Key:", type="password", key="openai_api_key")
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
            else:
                st.warning("⚠️ Please enter OpenAI API Key for LLM Recommendations")
            st.divider()

        interaction_data = tracker.get_interactions_data()
        
        if not interaction_data.empty:
            col1, col2, col3 = st.columns(3)
            total = len(interaction_data)
            clicks = len(interaction_data[interaction_data['interaction_score'] == 1])
            purchases = len(interaction_data[interaction_data['interaction_score'] == 2])
            
            col1.metric("Total", total)
            col2.metric("Clicks", clicks)
            col3.metric("Purchases", purchases)

            # Show progress towards auto-insertion
            if total < 50:
                st.progress(total/50, f"Collecting interactions ({total}/50 for auto-insertion)")
            
            st.dataframe(interaction_data, hide_index=True)
            
            if st.button("📥 Insert Interactions", key='insert_button', type="primary"):
                fg_updater.process_interactions(tracker, force=True)
        else:
            st.info("No interactions recorded yet")

def handle_llm_page(articles_fv, customer_id):
    """Handle LLM recommendations page"""
    if 'OPENAI_API_KEY' in os.environ:
        llm_recommendations(articles_fv, os.environ['OPENAI_API_KEY'], customer_id)
    else:
        st.warning("Please provide your OpenAI API Key in the Interaction Dashboard")

def process_pending_interactions(tracker, fg_updater):
    """Process pending interactions with proper notification"""
    interaction_count = len(tracker.get_interactions_data())
    
    if interaction_count >= 50:
        with st.spinner("🔄 Auto-inserting interactions..."):
            fg_updater.process_interactions(tracker, force=False)
        st.toast(f"✅ Auto-inserted {interaction_count} interactions", icon="📥")
        logger.info(f"Auto-inserted {interaction_count} interactions")
    else:
        fg_updater.process_interactions(tracker, force=False)

def main():
    # Initialize page
    initialize_page()
    
    # Initialize services
    tracker, fg_updater, articles_fv, ranking_deployment, query_model_deployment = initialize_services()
    
    # Select customer
    customer_id = st.sidebar.selectbox(
        '👤 Select Customer:',
        CUSTOMER_IDS,
        key='selected_customer'
    )
    
    # Page selection
    page_options = ["Customer Recommendations", "LLM Recommendations"]
    page_selection = st.sidebar.radio("📑 Choose Page:", page_options)
    
    # Process any pending interactions with notification
    process_pending_interactions(tracker, fg_updater)
    
    # Interaction dashboard with OpenAI API key field
    show_interaction_dashboard(tracker, fg_updater, page_selection)
    
    # Handle page content
    if page_selection == "Customer Recommendations":
        customer_recommendations(articles_fv, ranking_deployment, query_model_deployment, customer_id)
    else:  # LLM Recommendations
        handle_llm_page(articles_fv, customer_id)

if __name__ == '__main__':
    main()