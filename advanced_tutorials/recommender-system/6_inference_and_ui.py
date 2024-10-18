import streamlit as st
import hopsworks
from datetime import datetime
import pandas as pd
import numpy as np
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO
import os
import re
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import torch
from sentence_transformers import SentenceTransformer
from collections import defaultdict

# Function to print a styled header
def print_header(text, font_size=22):
    res = f'<span style="font-size: {font_size}px;">{text}</span>'
    st.markdown(res, unsafe_allow_html=True)

# Function to retrieve and start model deployments
@st.cache_resource()
def get_deployments():
    project = hopsworks.login()
    
    fs = project.get_feature_store()
    ms = project.get_model_serving()
    
    articles_fv = fs.get_feature_view(
        name="articles", 
        version=1,
    )

    query_model_deployment = ms.get_deployment("querydeployment")
    ranking_deployment = ms.get_deployment("rankingdeployment")
    
    ranking_deployment.start(await_running=180)
    query_model_deployment.start(await_running=180)
    
    return articles_fv, ranking_deployment, query_model_deployment

# Function to get item image URL
def get_item_image_url(item_id, articles_fv):
    return articles_fv.get_feature_vector({'article_id': item_id})[-1]

# Function to fetch and process image
@st.cache_data()
def fetch_and_process_image(image_url, width=200, height=300):
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        img = img.resize((width, height), Image.LANCZOS)
        return img
    except (UnidentifiedImageError, requests.RequestException, IOError):
        return None

def process_description(description):
    details_match = re.search(r'Details: (.+?)(?:\n|$)', description)
    return details_match.group(1) if details_match else "No details available."

def get_fashion_chain(api_key):
    model = ChatOpenAI(
        model_name='gpt-4o-mini-2024-07-18',
        temperature=0.7,
        openai_api_key=api_key,
    )
    template = """
    You are a fashion recommender for H&M. 
    
    Customer request: {user_input}
    
    Gender: {gender}
    
    Generate 3-5 necessary fashion items with detailed descriptions, tailored for an H&M-style dataset and appropriate for the specified gender. 
    Each item description should be specific, suitable for creating embeddings, and relevant to the gender. 
    
    STRICTLY FOLLOW the next response format:
    <emoji> <item 1 category> @ <item 1 description> | <emoji> <item 2 category> @ <item 2 description> | <emoji> <item 3 category> @ <item 3 description> | <Additional items if necessary> | <BRIEF OUTFIT SUMMARY AND STYLING TIPS WITH EMOJIS>
    
    Example for male gender:
    👖 Pants @ Slim-fit dark wash jeans with subtle distressing | 👕 Top @ Classic white cotton polo shirt with embroidered logo | 👟 Footwear @ Navy canvas sneakers with white soles | 🧥 Outerwear @ Lightweight olive green bomber jacket | 🕶️👔 Versatile casual look! Mix and match for various occasions. Add accessories for personal flair! 💼⌚
    
    Example for female gender:
    👗 Dress @ Floral print wrap dress with flutter sleeves | 👠 Footwear @ Strappy nude block heel sandals | 👜 Accessory @ Woven straw tote bag with leather handles | 🧥 Outerwear @ Cropped denim jacket with raw hem | 🌸👒 Perfect for a summer day out! Layer with the jacket for cooler evenings. Add a wide-brim hat for extra style! 💃🏻🕶️
    
    Ensure each item category has a relevant emoji, each item description is detailed, unique, and appropriate for the specified gender. 
    Make sure to take into account the gender when selecting items and descriptions. 
    The final section should provide a brief summary and styling tips with relevant emojis. Tailor your recommendations to the specified gender.
    """
    prompt = PromptTemplate(
        input_variables=["user_input", "gender"],
        template=template,
    )
    fashion_chain = LLMChain(
        llm=model,
        prompt=prompt,
        verbose=True
    )
    return fashion_chain

def get_fashion_recommendations(user_input, fashion_chain, gender):
    response = fashion_chain.run(user_input=user_input, gender=gender)
    items = response.strip().split(" | ")
    
    outfit_summary = items[-1] if len(items) > 1 else "No summary available."
    item_descriptions = items[:-1] if len(items) > 1 else items
    
    parsed_items = []
    for item in item_descriptions:
        try:
            emoji_category, description = item.split(" @ ", 1)
            emoji, category = emoji_category.split(" ", 1)
            parsed_items.append((emoji, category, description))
        except ValueError:
            # If splitting fails, use the whole item as description and use placeholders
            parsed_items.append(("🔷", "Item", item))
    
    return parsed_items, outfit_summary

def get_similar_items(item_description, embedding_model, articles_fv):
    item_description_embedding = embedding_model.encode(item_description)
    neighbors = articles_fv.find_neighbors(
        item_description_embedding, k=25,
    )
    return neighbors

def get_item_category(description):
    category_match = re.search(r'Category: (.+?) -', description)
    if category_match:
        return category_match.group(1)
    return 'Other'

def customer_recommendations(articles_fv, ranking_deployment, query_model_deployment):
    option_customer = st.sidebar.selectbox(
        'Select a customer:',
        (
            '641e6f3ef3a2d537140aaa0a06055ae328a0dddf2c2c0dd6e60eb0563c7cbba0',
            '1fdadbb8aa9910222d9bc1e1bd6fb1bd9a02a108cb0e899b640780f32d8f7d83',
            '7b0621c12c65570bdc4eadd3fca73f081e2da5769f0d31585ac301cea58af53f',
            '675cd49509ef9692d793af738c08d9bce0856036b9e988cba4e26422944314d6',
            '895576481a1095ad66ab3279483f4323724e9d53d9f089b16f289a3f660c1101',
        )
    )
    option_time = datetime.now().isoformat()
    
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = []
    if 'purchased_items' not in st.session_state:
        st.session_state.purchased_items = set()
    
    if st.sidebar.button('Get Recommendations') or not st.session_state.recommendations:
        st.write('🔮 Getting recommendations...')
        deployment_input = {
            "instances": {
                "customer_id": option_customer, 
                "transaction_date": option_time,
            }
        }
        prediction = query_model_deployment.predict(deployment_input)['predictions']['ranking']
        st.session_state.recommendations = [rec for rec in prediction if rec[1] not in st.session_state.purchased_items]
    
    print_header('📝 Top 12 Recommendations:')
    
    for row in range(3):
        cols = st.columns(4)
        for col in range(4):
            idx = row * 4 + col
            if idx < len(st.session_state.recommendations):
                recommendation = st.session_state.recommendations[idx]
                item_id = recommendation[1]
                score = recommendation[0]
                image_url = get_item_image_url(item_id, articles_fv)
                img = fetch_and_process_image(image_url)
                
                if img is None:
                    continue  # Skip this item if image couldn't be loaded
                
                full_description = articles_fv.get_feature_vector({'article_id': item_id})[-2]
                processed_description = process_description(full_description)
                
                with cols[col]:
                    st.image(img, use_column_width=True)
                    st.write(f"**🎯 Score:** {score:.4f}")
                    st.write("**📝 Details:**")
                    st.write(processed_description)
                    if st.button(f'🛒 Buy', key=f'buy_{item_id}'):
                        st.session_state.purchased_items.add(item_id)
                        st.success(f"✅ Item {item_id} purchased!")
                        st.session_state.recommendations = [
                            rec 
                            for rec 
                            in st.session_state.recommendations 
                            if rec[1] not in st.session_state.purchased_items
                        ]
                        st.experimental_rerun()

def llm_recommendations(articles_fv, api_key):
    st.write("🤖 LLM Fashion Recommender")
    
    gender = st.selectbox(
        "Select gender:",
        ("Male", "Female")
    )
    
    # Predefined options for user input
    input_options = [
        "I'm going to the beach for a week-long vacation. What items do I need?",
        "I have a formal winter wedding to attend next month. What should I wear?",
        "I'm starting a new job at a tech startup with a casual dress code. What items should I add to my wardrobe?",
        "Custom input"
    ]
    
    selected_input = st.selectbox(
        "Choose your fashion need or enter a custom one:",
        input_options
    )
    
    if selected_input == "Custom input":
        user_request = st.text_input("Enter your custom fashion need:")
    else:
        user_request = selected_input
    
    fashion_chain = get_fashion_chain(api_key)
    
    if 'llm_recommendations' not in st.session_state:
        st.session_state.llm_recommendations = []
    if 'purchased_items' not in st.session_state:
        st.session_state.purchased_items = set()
    if 'outfit_summary' not in st.session_state:
        st.session_state.outfit_summary = ""
    
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    if st.button("Get LLM Recommendations"):
        if not user_request:
            st.warning("Please enter a fashion need before generating recommendations.")
        else:
            with st.spinner("Generating recommendations..."):
                try:
                    item_recommendations, st.session_state.outfit_summary = get_fashion_recommendations(user_request, fashion_chain, gender)
                    st.session_state.llm_recommendations = []
                    
                    for emoji, category, item_description in item_recommendations:
                        similar_items = get_similar_items(item_description, embedding_model, articles_fv)
                        category_items = []
                        for similar_item in similar_items[:5]:
                            if similar_item[0] not in st.session_state.purchased_items:
                                category_items.append((item_description, similar_item))
                        if category_items:
                            st.session_state.llm_recommendations.append((emoji, category, category_items))
                    
                    if not st.session_state.llm_recommendations:
                        st.warning("No recommendations could be generated. Please try again or rephrase your request.")
                except Exception as e:
                    st.error(f"An error occurred while generating recommendations: {str(e)}")
    
    if st.session_state.outfit_summary:
        st.markdown(f"## 🎨 Outfit Summary")
        st.markdown(f"<h3 style='font-size: 30px;'>{st.session_state.outfit_summary}</h3>", unsafe_allow_html=True)
        st.markdown("---")
    
    # Display recommendations and handle purchases
    for emoji, category, items in st.session_state.llm_recommendations:
        st.markdown(f"## {emoji} {category}")
        if items:
            st.write(f"**Recommendation: {items[0][0]}**")
        
        cols = st.columns(5)
        items_to_remove = []
        for idx, (item_description, item) in enumerate(items):
            item_id = item[0]
            item_name = item[2]
            image_url = item[-1]
            description = process_description(item[-2])
            
            img = fetch_and_process_image(image_url)
            if img is None:
                continue  # Skip this item if image couldn't be loaded
            
            with cols[idx % 5]:
                st.image(img, use_column_width=True)
                st.write(f"**{item_name}**")
                st.write(description[:100] + "..." if len(description) > 100 else description)
                if st.button(f'🛒 Buy', key=f'buy_llm_{category}_{item_id}'):
                    st.session_state.purchased_items.add(item_id)
                    items_to_remove.append((item_description, item))
                    st.success(f"✅ Item {item_id} purchased!")
        
        # Remove purchased items and add new ones if necessary
        items = [item for item in items if item not in items_to_remove]
        if items_to_remove and items:
            new_items_needed = len(items_to_remove)
            similar_items = get_similar_items(items[0][0], embedding_model, articles_fv)
            for similar_item in similar_items:
                if new_items_needed == 0:
                    break
                if similar_item[0] not in st.session_state.purchased_items and (items[0][0], similar_item) not in items:
                    items.append((items[0][0], similar_item))
                    new_items_needed -= 1
        
        # Update the recommendations in the session state
        for i, (e, c, _) in enumerate(st.session_state.llm_recommendations):
            if e == emoji and c == category:
                st.session_state.llm_recommendations[i] = (emoji, category, items)
                break
        
        st.markdown("---")

    if st.button("Clear Purchases"):
        st.session_state.purchased_items.clear()
        st.success("All purchases have been cleared!")


def main():
    st.set_page_config(layout="wide", initial_sidebar_state='expanded')

    st.title('👒 Fashion Items Recommender')

    st.sidebar.title("⚙️ Configuration")
    
    with st.sidebar:
        st.write("🚀 Retrieving and Starting Deployments...")
        articles_fv, ranking_deployment, query_model_deployment = get_deployments()
        st.write('✅ Deployments are ready!')
        
        if st.button("Stop Deployments"):
            ranking_deployment.stop()
            query_model_deployment.stop()
            st.success("Deployments have been stopped!")
    
    page_options = ["Customer Recommendations", "LLM Recommendations"]
    page_selection = st.sidebar.radio("Choose a page", page_options)

    if page_selection == "LLM Recommendations":
        api_key = st.sidebar.text_input("🔑 Enter your OpenAI API Key:", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        else:
            st.sidebar.warning("⚠️ Please enter your OpenAI API Key to use LLM Recommendations.")

    if page_selection == "Customer Recommendations":
        customer_recommendations(articles_fv, ranking_deployment, query_model_deployment)
    elif page_selection == "LLM Recommendations":
        if 'OPENAI_API_KEY' in os.environ:
            llm_recommendations(articles_fv, os.environ['OPENAI_API_KEY'])
        else:
            st.warning("Please enter your OpenAI API Key in the sidebar to use LLM Recommendations.")

if __name__ == '__main__':
    main()