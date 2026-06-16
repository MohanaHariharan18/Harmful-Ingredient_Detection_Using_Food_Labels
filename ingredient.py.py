import streamlit as st
import cv2
import pytesseract
import pandas as pd
import re
import difflib
import numpy as np
import time
import tempfile
import os
from typing import List, Dict

class HarmfulIngredientDetector:
    def __init__(self, ingredients_data: pd.DataFrame = None, csv_path: str = None, excel_path: str = None):
        if ingredients_data is not None:
            self.ingredients_db = ingredients_data
        elif csv_path is not None:
            self.ingredients_db = pd.read_csv(csv_path)
        elif excel_path is not None:
            self.ingredients_db = pd.read_excel(excel_path)
        else:
            raise ValueError("Either ingredients_data, csv_path, or excel_path must be provided")

        self.ingredients_db['ingredient'] = self.ingredients_db['ingredient'].str.lower()
        self.ingredient_classes = {
            'harmful': set(self.ingredients_db[self.ingredients_db['class'] == 'harmful']['ingredient'].str.lower()),
            'controversial': set(self.ingredients_db[self.ingredients_db['class'] == 'controversial']['ingredient'].str.lower()),
            'not harmful': set(self.ingredients_db[self.ingredients_db['class'] == 'not harmful']['ingredient'].str.lower())
        }

    def _preprocess_ingredient(self, ingredient: str) -> str:
        cleaned = re.sub(r'\([^)]*\)', '', ingredient)
        cleaned = re.sub(r'[^a-zA-Z0-9\s-]', '', cleaned)
        return cleaned.strip().lower()

    def _find_direct_matches(self, ingredients: List[str]) -> List[Dict]:
        matches = []
        for ing in ingredients:
            cleaned = self._preprocess_ingredient(ing)
            classification = next((key for key, value in self.ingredient_classes.items() if cleaned in value), None)
            if classification:
                matches.append({
                    'original': ing,
                    'matched_as': cleaned,
                    'classification': classification,
                    'match_type': 'direct'
                })
        return matches
    
    def analyze_ingredients(self, ingredient_list: List[str]) -> Dict:
        results = { 'harmful': [], 'controversial': [], 'not harmful': [], 'unknown': [] }
        direct_matches = self._find_direct_matches(ingredient_list)
        categorized_ingredients = {match['original'] for match in direct_matches}

        for match in direct_matches:
            results[match['classification']].append(match)

        for ing in ingredient_list:
            if ing not in categorized_ingredients:
                results['unknown'].append({'original': ing})

        return results

    def analyze_from_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        extracted_text = pytesseract.image_to_string(gray)
        ingredient_list = re.findall(r'[A-Za-z]+(?:[-\s][A-Za-z]+)*', extracted_text)
        return ingredient_list, self.analyze_ingredients(ingredient_list)

    def analyze_from_text(self, text):
        ingredient_list = [ing.strip() for ing in text.split(',')]
        return ingredient_list, self.analyze_ingredients(ingredient_list)

# Streamlit app
st.set_page_config(
    page_title="Harmful Ingredient Detector",
    page_icon="🔍",
    layout="wide"
)

st.title("Harmful Ingredient Detector")
st.write("Scan product labels or enter ingredients manually to detect harmful or controversial ingredients.")

# Sidebar for app navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose Input Method:", ["Upload Image", "Camera Capture", "Manual Entry"])

# Load ingredient database
@st.cache_data
def load_data():
    try:
        return pd.read_csv("ing_data.csv")
    except FileNotFoundError:
        st.error("Ingredient database (ing_data.csv) not found. Using sample data instead.")
        # Create a sample dataframe if the file is not found
        return pd.DataFrame({
            'ingredient': ['Sodium Lauryl Sulfate', 'Parabens', 'Glycerin', 'Fragrance', 'Aloe Vera'],
            'class': ['harmful', 'controversial', 'not harmful', 'controversial', 'not harmful']
        })

ingredients_data = load_data()
detector = HarmfulIngredientDetector(ingredients_data=ingredients_data)

# Function to display results
def display_results(ingredient_list, results):
    st.subheader("Extracted Ingredients")
    st.write(", ".join(ingredient_list))
    
    st.subheader("Analysis Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("### Harmful ⚠️")
        if results['harmful']:
            for item in results['harmful']:
                st.markdown(f"- **{item['original']}**")
        else:
            st.write("None found")
    
    with col2:
        st.markdown("### Controversial ⚖️")
        if results['controversial']:
            for item in results['controversial']:
                st.markdown(f"- **{item['original']}**")
        else:
            st.write("None found")
    
    with col3:
        st.markdown("### Not Harmful ✅")
        if results['not harmful']:
            for item in results['not harmful']:
                st.markdown(f"- **{item['original']}**")
        else:
            st.write("None found")
    
    with col4:
        st.markdown("### Unknown ❓")
        if results['unknown']:
            for item in results['unknown']:
                st.markdown(f"- **{item['original']}**")
        else:
            st.write("None found")
    
    # Summary statistics
    st.subheader("Summary")
    total = len(ingredient_list)
    harmful_count = len(results['harmful'])
    controversial_count = len(results['controversial'])
    safe_count = len(results['not harmful'])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Ingredients", total)
    col2.metric("Harmful", harmful_count, f"{harmful_count/total*100:.1f}%" if total > 0 else "0%")
    col3.metric("Controversial", controversial_count, f"{controversial_count/total*100:.1f}%" if total > 0 else "0%")
    col4.metric("Safe", safe_count, f"{safe_count/total*100:.1f}%" if total > 0 else "0%")

# Different app modes
if app_mode == "Upload Image":
    st.header("Upload Product Label Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Process the image
        file_bytes = uploaded_file.getvalue()
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Analyze Ingredients"):
            with st.spinner("Analyzing image..."):
                ingredient_list, results = detector.analyze_from_image(image)
                display_results(ingredient_list, results)

elif app_mode == "Camera Capture":
    st.header("Capture Product Label with Camera")
    
    # Start the camera
    camera_image = st.camera_input("Take a picture of the product label")
    
    if camera_image is not None:
        # Convert to OpenCV format and process
        file_bytes = camera_image.getvalue()
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if st.button("Analyze Ingredients"):
            with st.spinner("Analyzing image..."):
                ingredient_list, results = detector.analyze_from_image(image)
                display_results(ingredient_list, results)

else:  # Manual Entry
    st.header("Enter Ingredients Manually")
    ingredients_text = st.text_area(
        "Enter ingredients (comma separated):",
        "Water, Glycerin, Sodium Lauryl Sulfate, Fragrance, Aloe Vera",
        height=150
    )
    
    if st.button("Analyze Ingredients"):
        with st.spinner("Analyzing ingredients..."):
            ingredient_list, results = detector.analyze_from_text(ingredients_text)
            display_results(ingredient_list, results)

# Add database management section in the sidebar
st.sidebar.markdown("---")
st.sidebar.header("Database Management")

# Option to view current database
if st.sidebar.checkbox("View Ingredient Database"):
    st.sidebar.dataframe(ingredients_data)

# Import missing libraries
import numpy as np

# Run this with: streamlit run app.py