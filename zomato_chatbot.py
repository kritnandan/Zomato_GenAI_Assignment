import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import numpy as np
import json

@st.cache_resource
#loading the dietary mapping which already mapped in the JSON file
def load_dietary_mapping():
    with open('dietary_mapping.JSON', 'r') as f:
        return json.load(f)
# Loding the Dataset
@st.cache_data
def load_data():
    df = pd.read_csv('zomato_full_data.csv')
    
# Defined the Keywords for Veg and Non-Veg
    veg_keywords = [
        'veg', 'vegetarian', 'paneer', 'aloo', 'dal', 'bhature', 'puri', 'idli', 'dosa',
        'palak', 'gobi', 'mushroom', 'malai kofta', 'rajma', 'chole', 'bhindi',
        'sabzi', 'kadhai paneer', 'shahi paneer', 'matar', 'naan', 'roti',
        'paratha', 'pulao', 'rice', 'biryani veg', 'veg biryani', 'samosa',
        'pakora', 'pavbhaji', 'pav bhaji', 'thali veg', 'dahi', 'raita',
        'khichdi', 'upma', 'uttapam', 'sambhar', 'rasam'
    ]
    
    non_veg_keywords = [
        'chicken', 'mutton', 'egg', 'fish', 'meat', 'kebab', 'tikka',
        'tandoori chicken', 'butter chicken', 'biryani chicken', 'chicken biryani',
        'fish curry', 'prawn', 'seafood', 'mutton biryani', 'keema',
        'chicken tikka', 'chicken 65', 'chicken masala', 'egg curry',
        'chicken curry', 'mutton curry', 'fish fry', 'chicken wings',
        'chicken roll', 'egg roll', 'chicken burger', 'non veg thali'
    ]
    

    df['Category'] = 'Unknown'
    
# Function to categorize dishes
    def categorize_dish(dish_name):
        if pd.isna(dish_name):
            return 'Unknown'
        
        dish_lower = str(dish_name).lower()
        
# Check for explicit vegetarian/non-vegetarian mentions
        if 'non-veg' in dish_lower or 'non veg' in dish_lower:
            return 'Non-Vegetarian'
        if 'veg' in dish_lower and 'non' not in dish_lower:
            return 'Vegetarian'
        
# Check keywords
        if any(keyword in dish_lower for keyword in non_veg_keywords):
            return 'Non-Vegetarian'
        if any(keyword in dish_lower for keyword in veg_keywords):
            return 'Vegetarian'
        
# Handle special cases
        if 'combo' in dish_lower or 'meal' in dish_lower:
            if any(keyword in dish_lower for keyword in non_veg_keywords):
                return 'Non-Vegetarian'
            return 'Vegetarian'  # Default combos to veg unless specified
        
# Handle thalis
        if 'thali' in dish_lower:
            if any(keyword in dish_lower for keyword in non_veg_keywords):
                return 'Non-Vegetarian'
            return 'Vegetarian'
        
        return 'Unknown'
    
# Apply categorization
    df['Category'] = df['Dish Name'].apply(categorize_dish)
    
# Add confidence score for categorization
    def get_confidence_score(dish_name):
        if pd.isna(dish_name):
            return 0.0
        
        dish_lower = str(dish_name).lower()
        veg_matches = sum(1 for keyword in veg_keywords if keyword in dish_lower)
        non_veg_matches = sum(1 for keyword in non_veg_keywords if keyword in dish_lower)
        
        if veg_matches > 0 and non_veg_matches == 0:
            return min(1.0, 0.7 + (veg_matches * 0.1))
        elif non_veg_matches > 0 and veg_matches == 0:
            return min(1.0, 0.7 + (non_veg_matches * 0.1))
        elif veg_matches == 0 and non_veg_matches == 0:
            return 0.5
        else:
# If both veg and non-veg keywords found, mark as uncertain
            return 0.3
    
    df['Category_Confidence'] = df['Dish Name'].apply(get_confidence_score)
    
    return df

# Initialize models
@st.cache_resource
def load_embed_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_generator():
    return pipeline('text-generation', 
                   model='gpt2',
                   device='cpu')

# Prepare contexts with category information
@st.cache_data
def prepare_contexts(df):
    contexts = []
    for _, group in df.groupby('Restaurant'):
        context = f"Restaurant: {group['Restaurant'].iloc[0]}\n"
        context += f"Address: {group['Address'].iloc[0]}\n"
        context += f"Status: {group['Status'].iloc[0]}\n"
        context += f"Timings: {group['Timings'].iloc[0]}\n"
        context += f"Contact: {group['Contact'].iloc[0]}\n"
        
# Separate veg and non-veg items
        veg_items = group[group['Category'] == 'Vegetarian']
        non_veg_items = group[group['Category'] == 'Non-Vegetarian']
        
        if not veg_items.empty:
            context += "\nVegetarian Menu Items:\n"
            for _, row in veg_items.iterrows():
                if pd.notna(row['Dish Name']):
                    item = f"- {row['Dish Name']}"
                    if pd.notna(row['Price']):
                        item += f" (Price: {row['Price']})"
                    context += item + "\n"
        
        if not non_veg_items.empty:
            context += "\nNon-Vegetarian Menu Items:\n"
            for _, row in non_veg_items.iterrows():
                if pd.notna(row['Dish Name']):
                    item = f"- {row['Dish Name']}"
                    if pd.notna(row['Price']):
                        item += f" (Price: {row['Price']})"
                    context += item + "\n"
        
        contexts.append(context)
    return contexts

@st.cache_resource
def compute_embeddings(_embed_model, contexts):
    return _embed_model.encode(contexts)

def retrieve_info(query, embed_model, embeddings, contexts, top_k=3):
    query_embedding = embed_model.encode(query)
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [contexts[i] for i in top_indices]

def generate_response(query, retrieved_info, generator, conversation_history):
    prompt = "You are a helpful restaurant assistant. Answer the user's question based on the following information:\n\n"
    
    for info in retrieved_info:
        prompt += f"Information:\n{info}\n\n"
    
    prompt += f"Conversation History:\n{conversation_history}\n\n" if conversation_history else ""
    prompt += f"Question: {query}\nAnswer:"
    
    response = generator(
        prompt,
        max_new_tokens=200,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=generator.tokenizer.eos_token_id
    )
    
    generated_text = response[0]['generated_text']
    answer = generated_text.replace(prompt, "").strip()
    
    if '.' in answer:
        answer = answer[:answer.rfind('.')+1]
    
    return answer

# Query handling function

def handle_query(query, df, embed_model, embeddings, contexts, generator, conversation_history):
    query_lower = query.lower()

#  1. Contact Info Retrieval
    if "contact" in query_lower or "phone" in query_lower or "number" in query_lower:
        for restaurant in df['Restaurant'].unique():
            if restaurant.lower() in query_lower:
                contact = df[df['Restaurant'] == restaurant]['Contact'].iloc[0]
                return f"📞 The contact number for **{restaurant}** is: **{contact}**"
        return "❌ I couldn't find the contact number for that restaurant. Please make sure the name is correct."

# 2. Timing Information 
    if any(word in query_lower for word in ["timing", "hours", "open", "close", "when"]):
        for restaurant in df['Restaurant'].unique():
            if restaurant.lower() in query_lower:
                timings = df[df['Restaurant'] == restaurant]['Timings'].iloc[0]
                if pd.notna(timings):
                    timings = timings.split('(')[0].strip()
                    return f"⏰ **{restaurant}** is open:\n**{timings}**"
                else:
                    return f"❌ Sorry, timing information is not available for **{restaurant}**."
        return "❌ Please specify a restaurant name to check timings. For example: 'What are the timings of Paradise?'"

# 3. Price Query
    if "price" in query_lower:
        dish_found = None
        dish_names = sorted(df['Dish Name'].dropna().unique(), key=len, reverse=True)
        for dish in dish_names:
            dish_words = set(dish.lower().split())
            query_words = set(query_lower.split())
            if dish_words.issubset(query_words):
                dish_found = dish
                break
        
        if dish_found:
 # Find all restaurants serving this dish
            matches = df[df['Dish Name'].str.lower() == dish_found.lower()]
            if matches.empty:
                return f"❌ I couldn't find **{dish_found}** in any restaurant."

            if len(matches) == 1:
# Single restaurant case
                row = matches.iloc[0]
                price = str(row['Price']).replace('₹', '').strip() if pd.notna(row['Price']) else None
                restaurant = row['Restaurant']
                if price:
                    return f"💰 **{dish_found}** is available at **{restaurant}** for ₹{price}"
                else:
                    return f"❌ Price not available for **{dish_found}** at **{restaurant}**"
            else:
# Multiple restaurants case
                response = f"💰 **{dish_found}** is available at:\n\n"
                for _, row in matches.iterrows():
                    restaurant = row['Restaurant']
                    price = str(row['Price']).replace('₹', '').strip() if pd.notna(row['Price']) else None
                    if price:
                        response += f"- **{restaurant}**: ₹{price}\n"
                    else:
                        response += f"- **{restaurant}**: Price not available\n"
                return response
        
        return "❌ Please specify a dish name clearly. For example: 'What is the price of Badam Lachha?'"

# 4. Menu Listing 
    if "menu" in query_lower:
        for restaurant in df['Restaurant'].unique():
            if restaurant.lower() in query_lower:
                items = df[df['Restaurant'] == restaurant][['Dish Name', 'Price']]
                if items.empty:
                    return f"❌ No menu items found for **{restaurant}**."
                response = f"📋 **Menu for {restaurant}:**\n\n"
                for _, row in items.iterrows():
                    dish = row['Dish Name']
                    price = row['Price']
                    if pd.notna(price):
                        price = str(price).replace('₹', '').strip()
                        price = f"(₹{price})"
                    else:
                        price = "(Price not available)"
                    response += f"- {dish} {price}\n"
                return response
        return "❌ Couldn't find the restaurant in our database."

#  5. Dish Availability in a Restaurant 
    if "available" in query_lower or "do they have" in query_lower:
        for dish in df['Dish Name'].dropna().unique():
            if dish.lower() in query_lower:
                for restaurant in df['Restaurant'].unique():
                    if restaurant.lower() in query_lower:
                        match = df[(df['Dish Name'].str.lower() == dish.lower()) & 
                                 (df['Restaurant'].str.lower() == restaurant.lower())]
                        if not match.empty:
                            price = str(match['Price'].values[0]).replace('₹', '').strip() if pd.notna(match['Price'].values[0]) else None
                            if price:
                                price_info = f" The price is ₹{price}."
                            else:
                                price_info = " The price is not available."
                            return f"✅ Yes, **{restaurant}** offers **{dish}**.{price_info}"
                        else:
                            return f"❌ No, **{restaurant}** doesn't seem to offer **{dish}**."
        return "❌ I couldn't identify the dish or restaurant. Could you rephrase?"
#  6. Veg / Non-Veg Classification at Restaurant Level 
    if "vegetarian" in query_lower or "non-vegetarian" in query_lower or "veg" in query_lower or "non veg" in query_lower:
        veg_only = []
        nonveg_only = []
        mixed = []

        for restaurant in df['Restaurant'].unique():
            rest_data = df[df['Restaurant'] == restaurant]
            has_veg = not rest_data[rest_data['Category'] == 'Vegetarian'].empty
            has_nonveg = not rest_data[rest_data['Category'] == 'Non-Vegetarian'].empty

            if has_veg and not has_nonveg:
                veg_only.append(restaurant)
            elif has_nonveg and not has_veg:
                nonveg_only.append(restaurant)
            elif has_veg and has_nonveg:
                mixed.append(restaurant)

        if "non" in query_lower:
            if nonveg_only or mixed:
                response = "🍗 **Restaurants with Non-Vegetarian Options:**\n"
                if mixed:
                    response += "- " + "\n- ".join([f"{r} (also serves veg)" for r in mixed]) + "\n"
                if nonveg_only:
                    response += "- " + "\n- ".join(nonveg_only)
            else:
                response = "❌ Couldn't find restaurants with non-vegetarian options."
            return response

        elif "veg" in query_lower or "vegetarian" in query_lower:
            if veg_only or mixed:
                response = "🥦 **Restaurants with Vegetarian Options:**\n"
                if mixed:
                    response += "- " + "\n- ".join([f"{r} (also serves non-veg)" for r in mixed]) + "\n"
                if veg_only:
                    response += "- " + "\n- ".join(veg_only)
            else:
                response = "❌ Couldn't find restaurants with vegetarian options."
            return response

# 7. Dietary Restrictions 
    query_lower = query.lower()
    dietary_mapping = load_dietary_mapping()

# Handle spice level queries
    if any(word in query_lower for word in ["spicy", "spice", "mild"]):
        if not any(level.replace("_", " ") in query_lower for level in ["low_spice", "medium_spice", "high_spice"]):
            return ("🌶️ Please specify your preferred spice level:\n\n"
                   "🥗 Low Spice (Mild)\n"
                   "🌶️ Medium Spice\n"
                   "🔥 High Spice")

        for level in ["low_spice", "medium_spice", "high_spice"]:
            if level.replace("_", " ") in query_lower:
                dishes = dietary_mapping["dietary_categories"]["spice_level"][level]
                return format_dish_response(df, dishes, f"{level.replace('_', ' ')} dishes", "🌶️")

# Handle dietary preferences (vegetarian, vegan, jain)
    for diet_type in ["vegetarian", "vegan", "jain"]:
        if any(kw in query_lower for kw in dietary_mapping["dietary_categories"][diet_type]["keywords"]):
            dishes = dietary_mapping["dietary_categories"][diet_type]["dishes"]
            confidence = dietary_mapping["dietary_categories"][diet_type]["confidence_level"]
            return format_dish_response(df, dishes, f"{diet_type} dishes", "🥗", confidence)

# Handle pizza base preferences
    if "pizza" in query_lower and any(base in query_lower for base in ["thin", "thick", "crust"]):
        base_type = "thin_crust" if "thin" in query_lower else "thick_crust"
        pizzas = dietary_mapping["dietary_categories"]["pizza_base"][base_type]
        return format_dish_response(df, pizzas, f"{base_type.replace('_', ' ')} pizzas", "🍕")

# Handle combo meals
    if "combo" in query_lower or "family" in query_lower or "pack" in query_lower:
        combo_type = "family_pack" if any(w in query_lower for w in ["family", "group"]) else "single_serve"
        combos = dietary_mapping["dietary_categories"]["combos"][combo_type]
        return format_dish_response(df, combos, f"{combo_type.replace('_', ' ')} meals", "🍱")

# Handle allergen information
    if any(allergen in query_lower for allergen in ["dairy", "gluten", "soy"]) or "allergy" in query_lower:
        response = "⚠️ **Allergen Information:**\n\n"
        for allergen, items in dietary_mapping["allergens"].items():
            if allergen in query_lower or "allergy" in query_lower:
                response += f"🚫 **{allergen.title()} containing items:**\n"
                for item in items:
                    response += f"- {item}\n"
                response += "\n"
        return response + "*Note: Always confirm allergen information with the restaurant.*"

# Handle preparation style queries
    if any(style in query_lower for style in ["fried", "baked", "steamed", "grilled"]):
        for style, items in dietary_mapping["preparation_style"].items():
            if style in query_lower:
                return format_dish_response(df, items, f"{style} dishes", "👨‍🍳")

def format_dish_response(df, dishes, category_name, emoji, confidence=None):
    """Helper function to format dish responses"""
    matched_items = df[df['Dish Name'].isin(dishes)]
    if matched_items.empty:
        return f"❌ No {category_name} found in our database."

    response = f"{emoji} Here are {category_name}"
    if confidence:
        response += f" (Confidence: {confidence})"
    response += ":\n\n"

    for _, row in matched_items.iterrows():
        price = str(row['Price']).replace('₹', '').strip() if pd.notna(row['Price']) else "Price not available"
        response += f"- **{row['Dish Name']}** at *{row['Restaurant']}* (₹{price})\n"

    if confidence == "medium":
        response += "\n💡 *Note: Please confirm dietary requirements with the restaurant.*"
    
    return response
#  8. Restaurant Location with Google Maps Link 
    if "where" in query_lower or "location" in query_lower or "located" in query_lower or "address" in query_lower:
        for restaurant in df['Restaurant'].unique():
            if restaurant.lower() in query_lower:
                location = df[df['Restaurant'] == restaurant]['Address'].iloc[0]
                maps_query = f"{restaurant}, {location}".replace(' ', '+')
                maps_url = f"https://www.google.com/maps/search/?api=1&query={maps_query}"
                return f"📍 **{restaurant}** is located at: **{location}**\n\n🗺️ [View on Google Maps]({maps_url})"
        return "❌ I couldn't find the location for that restaurant."

    return "🤔 I'm not sure how to help with that yet. Could you please rephrase your question?"
# 9. Fallback 
    return "🤖 I'm not sure how to help with that yet. Could you try rephrasing your question?"
     
# Use RAG pipeline to handle everything else
    retrieved_info = retrieve_info(query, embed_model, embeddings, contexts)
    response = generate_response(query, retrieved_info, generator, conversation_history)
    return response if response else "❌ Sorry, I couldn't understand that. Can you try rephrasing?"

def main():
# Custom CSS for better styling
    st.markdown("""
        <style>
        .zomato-logo {
            color: #CB202D;
            font-size: 2.5rem;
            margin-bottom: 1rem
        }
        </style>
        """, unsafe_allow_html=True)
    st.markdown("<h1 class='zomato-logo'>Zomato Chatbot</h1>""", unsafe_allow_html=True)
    
      
    st.markdown("""
    ### Hi there! I'm your friendly Zomato assistant 👋
    I can help you discover amazing restaurants, find your favorite dishes, and more!
    """)

# Loading data and models
    df = load_data()
    embed_model = load_embed_model()
    generator = load_generator()
    
# Preparing contexts and embeddings
    contexts = prepare_contexts(df)
    embeddings = compute_embeddings(embed_model, contexts)
    
# Initialize conversation history
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    
# Enhanced sidebar with better categorization
    with st.sidebar:
        st.header("💡 What Can I Help You With?")
        
        st.subheader("🔍 Find Restaurants")
        st.markdown("""
        - Where can I find the best biryani?
        - Show me vegetarian restaurants
        - What restaurants are near me?
        """)
        
        st.subheader("🍜 Menu & Prices")
        st.markdown("""
        - What's on the menu at Paradise?
        - How much is Butter Chicken at Mehfil?
        - Compare prices for Biryani
        """)
        
        st.subheader("🥗 Dietary Preferences")
        st.markdown("""
        - Show me gluten-free options
        - Which places have good vegan food?
        - Are there any low-calorie dishes?
        """)

# Chat interface with better visual separation
    st.markdown("---")
    
# Display conversation with improved styling
    for exchange in st.session_state.conversation:
        with st.chat_message(exchange["role"], avatar="👤" if exchange["role"] == "user" else "🤖"):
            st.markdown(exchange["content"])
    
# More inviting input prompt
    user_query = st.chat_input("What would you like to know about food & restaurants? 😋")
    
    if user_query:
        st.session_state.conversation.append({"role": "user", "content": user_query})
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_query)
        
        with st.spinner("🔍 Looking for the perfect answer..."):
            conversation_history = "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in st.session_state.conversation[-4:]]
            )
            
            response = handle_query(
                user_query, 
                df, 
                embed_model, 
                embeddings, 
                contexts, 
                generator, 
                conversation_history
            )
        
        st.session_state.conversation.append({"role": "assistant", "content": response})
        
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(response)

   

if __name__ == "__main__":
    main()