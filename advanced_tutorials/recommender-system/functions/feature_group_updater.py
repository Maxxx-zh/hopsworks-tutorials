import pandas as pd
from datetime import datetime
import streamlit as st
import hopsworks
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureGroupUpdater:
    def __init__(self, batch_size: int = 50):
        """Initialize the FeatureGroup updater.
        
        Args:
            batch_size: Number of interactions before automatic insertion
        """
        self.batch_size = batch_size
        self._initialize_feature_group()
    
    def _initialize_feature_group(self) -> None:
        """Initialize connection to Hopsworks Feature Group"""
        try:
            if 'feature_group' not in st.session_state:
                logger.info("📡 Initializing Hopsworks Feature Group connection...")
                project = hopsworks.login()
                fs = project.get_feature_store()
                st.session_state.feature_group = fs.get_feature_group(
                    name="interactions",
                    version=1,
                )
                logger.info("✅ Feature Group connection established")

            # Initialize last processed timestamp if not exists
            if 'last_processed_timestamp' not in st.session_state:
                st.session_state.last_processed_timestamp = int(datetime.now().timestamp())
                logger.info(f"⛳️ Initialized last_processed_timestamp: {st.session_state.last_processed_timestamp}")

        except Exception as e:
            logger.error(f"Failed to initialize Feature Group connection: {str(e)}")
            st.error("❌ Failed to connect to Feature Group. Check terminal for details.")
            raise

    def _prepare_interactions_for_insertion(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare interactions dataframe to match Feature Group schema"""
        if df is None or df.empty:
            return None

        try:
            # Create a new DataFrame with required columns and types
            prepared_df = pd.DataFrame({
                't_dat': df['t_dat'].astype('int64'),
                'customer_id': df['customer_id'].astype(str),
                'article_id': df['article_id'].astype(str),
                'interaction_score': df['interaction_score'].astype('int64'),
                'prev_article_id': df['prev_article_id'].astype(str)
            })

            logger.info(f"Prepared {len(prepared_df)} interactions for insertion")
            return prepared_df

        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            st.error("❌ Failed to prepare data. Check terminal for details.")
            return None

    def _get_new_interactions(self, tracker) -> Optional[pd.DataFrame]:
        """Get new interactions since last processing"""
        try:
            interactions_df = tracker.get_interactions_data()
            
            if interactions_df.empty:
                logger.info("No interactions found")
                return None

            # Convert t_dat to Unix timestamp if needed
            if not pd.api.types.is_integer_dtype(interactions_df['t_dat']):
                interactions_df['t_dat'] = pd.to_datetime(interactions_df['t_dat']).astype('int64') // 10**9

            # Filter new interactions
            new_interactions = interactions_df[
                interactions_df['t_dat'] > st.session_state.last_processed_timestamp
            ]

            if new_interactions.empty:
                logger.info("No new interactions found")
                return None

            logger.info(f"Found {len(new_interactions)} new interactions")
            return self._prepare_interactions_for_insertion(new_interactions)

        except Exception as e:
            logger.error(f"Error getting new interactions: {str(e)}")
            st.error("❌ Failed to get new interactions. Check terminal for details.")
            return None

    def insert_interactions(self, new_interactions: pd.DataFrame) -> bool:
        """Insert interactions into Feature Group"""
        try:
            if new_interactions is not None and not new_interactions.empty:
                n_interactions = len(new_interactions)
                logger.info(f"Inserting {n_interactions} interactions...")
                
                with st.spinner(f"Inserting {n_interactions} interactions..."):
                    st.session_state.feature_group.insert(new_interactions)
                    st.session_state.last_processed_timestamp = int(new_interactions['t_dat'].max())
                
                logger.info("Insertion completed successfully")
                return True

        except Exception as e:
            logger.error(f"Failed to insert interactions: {str(e)}")
            st.error("❌ Failed to insert interactions. Check terminal for details.")

        return False
    
    def process_interactions(self, tracker, force: bool = False) -> bool:
        """Process interactions from the tracker and insert into Feature Group"""
        try:
            # Get new interactions
            new_interactions = self._get_new_interactions(tracker)

            if new_interactions is None:
                if force:
                    logger.info("ℹ️ No new interactions to insert")
                return False

            # Insert if forced or batch size reached
            if force or len(new_interactions) >= self.batch_size:
                logger.info("Starting insertion...")
                if self.insert_interactions(new_interactions):
                    logger.info(f"✅ Successfully inserted {len(new_interactions)} interactions")
                    return True
                else:
                    logger.error("Failed to insert interactions")
                    return False

        except Exception as e:
            logger.error(f"Error processing interactions: {str(e)}")
            return False

        return False

def get_fg_updater(batch_size: int = 50):
    """Get or create FeatureGroupUpdater instance"""
    if 'fg_updater' not in st.session_state:
        st.session_state.fg_updater = FeatureGroupUpdater(batch_size=batch_size)
    return st.session_state.fg_updater