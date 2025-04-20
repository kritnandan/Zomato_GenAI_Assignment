# 🍽️ Zomato Restaurant Chatbot


A Streamlit-powered chatbot app that answers user queries about restaurant menus, dishes, dietary preferences, prices, and locations using Retrieval-Augmented Generation (RAG) and structured menu data.

---

## Features

- Find restaurant addresses with Google Maps link  
- Get contact information  
- Show operating hours of restaurants  
- Filter dishes by dietary preferences (vegan, gluten-free, Jain, etc.)  
- Get info on spice levels, allergens, and pizza base  
- Check dish availability and compare prices  
- Veg/Non-Veg classification with confidence scores  
- Handles out-of-scope or unclear questions intelligently 

---

##  Setup Instructions

### 1. Create a Virtual Environment (Recommended)

```bash
python -m venv chatbot_env
source chatbot_env/bin/activate  # For Windows: chatbot_env\Scripts\activate
````
# Web Scraping: Install Chrome WebDriver
- Download Chrome WebDriver that matches your Chrome browser version.
- Place the driver in the project root folder or add its path to your system environment variables


# Install Required Packages
- pip install -r requirements.txt

#  Requirements
- Python 3.8+
- Streamlit
- Pandas
- Scikit-learn
- Selenium (for scraping)
- BeautifulSoup
- Chrome WebDriver

#  Notes
- zomato_full_data.csv contains columns like:
- Restaurant Name, Dish, Price, Address, Timings, ContactThe chatbot uses keyword-based logic and lightweight retrieval techniques to handle various types of queries.

# Dietary filtering is powered by dietary_mapping.json, which supports:

- Spice level
- Dietary preferences (vegan, Jain, etc.)
- Allergens
- Pizza base types
- Combo types
- Cooking styles

# Instructions for running the code 
- Create the virtual environment
- Download the WebDriver for scraping 
- Run the web scraping code to scrape data from the restaurant's webpage
- Save the data in CSV format
- Download zomato_chatbot.py in the same folder
- Run requirements.txt first
- Download dietary_mapping.json in the same folder
- Run with streamlit run zomato_chatbot.py



