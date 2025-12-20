import streamlit as st
import pandas as pd
import joblib
import os

# Set Page Config
st.set_page_config(
    page_title="BMW Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# Load Model
@st.cache_resource
def load_model():
    if os.path.exists("bmw_price_model.pkl"):
        return joblib.load("bmw_price_model.pkl")
    else:
        return None

artifacts = load_model()

# Title and Description
st.title("🚗 BMW Used Car Price Predictor")
st.markdown("""
Welcome to the **BMW Price Prediction System**. 
This tool uses Machine Learning to estimate the market price of a used BMW based on its features.

**Instructions:**
1. Enter the car details in the form below.
2. Click **'Predict Price'**.
3. (Optional) Provide feedback on the prediction.
""")

if artifacts is None:
    st.error("Error: Model file 'bmw_price_model.pkl' not found. Please run Bmw.py first.", icon="🚨")
    st.stop()

model = artifacts["model"]
encoders = artifacts["encoders"]
feature_names = artifacts["features"]
r2_score = artifacts["r2_score"]

# Sidebar - About
with st.sidebar:
    st.header("About Model")
    st.info(f"Using Algorithm: **{artifacts['model_name']}**")
    st.success("System Status: **Online**")
    st.markdown("---")
    st.write("Developed for **Data Science Assignment 04**")

st.markdown("---")
st.header("🚘 Car Details")

# Input Form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        # We need to know the options for categorical features. 
        # Since we have the encoders, we can get the classes.
        
        # Model
        model_options = list(encoders['Model'].classes_)
        selected_model = st.selectbox("Model", model_options)
        
        # Year
        selected_year = st.number_input("Year", min_value=1990, max_value=2024, value=2018, step=1)
        
        # Mileage
        mileage = st.number_input("Mileage (KM)", min_value=0, max_value=500000, value=50000, step=1000)
    
    with col2:
        # Fuel Type
        fuel_options = list(encoders['Fuel_Type'].classes_)
        selected_fuel = st.selectbox("Fuel Type", fuel_options)
        
        # Transmission
        trans_options = list(encoders['Transmission'].classes_)
        selected_trans = st.selectbox("Transmission", trans_options)
        
        # Engine Size
        engine_size = st.number_input("Engine Size (L)", min_value=1.0, max_value=6.0, value=2.0, step=0.1)

    # Hidden/Default inputs for other required features (Region, Color, Sales_Class)
    # For a simple demo, we can set defaults or add them to UI. Let's add them to "Advanced" expander.
    
    with st.expander("Advanced Features (Optional)"):
        c1, c2, c3 = st.columns(3)
        region_opts = list(encoders['Region'].classes_)
        selected_region = c1.selectbox("Region", region_opts)
        
        color_opts = list(encoders['Color'].classes_)
        selected_color = c2.selectbox("Color", color_opts)
        
        type_opts = list(encoders['Sales_Classification'].classes_)
        selected_type = c3.selectbox("Sales Type", type_opts)

    submit_btn = st.form_submit_button("💰 Predict Price", type="primary")

if submit_btn:
    # Prepare Input Data
    # 1. Create DataFrame with single row
    input_data = pd.DataFrame({
        'Model': [selected_model],
        'Year': [selected_year],
        'Price_USD': [0], # Dummy placeholder
        'Transmission': [selected_trans],
        'Mileage_KM': [mileage],
        'Fuel_Type': [selected_fuel],
        'Region': [selected_region],
        'Color': [selected_color],
        'Engine_Size_L': [engine_size],
        'Sales_Classification': [selected_type]
    })
    
    # 2. Re-order columns to match training features exactly
    # But wait, 'Price_USD' was the target, it shouldn't be in features.
    # The 'features' list in artifacts will tell us what X had.
    
    # We need to construct the exact feature vector X for model input.
    # X_train features were used. Let's filter input_data to those features.
    
    # Encode Categoricals using saved encoders
    try:
        encoded_input = input_data.copy()
        for col, le in encoders.items():
            val = encoded_input[col].iloc[0]
            encoded_input[col] = le.transform([val])
            
        # Select only feature columns
        final_input = encoded_input[feature_names]
        
        # Predict
        prediction = model.predict(final_input)[0]
        
        st.markdown("---")
        st.subheader("Prediction Result")
        st.success(f"### Estimated Price: **${prediction:,.2f}**")
        
        if prediction < 0:
            st.warning("Note: The model predicted a negative value, which is unrealistic. This model might need more training data.")
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")


# ==========================================
# Task 2: Feedback Collection
# ==========================================
st.markdown("---")
st.header("📝 Feedback Collection")
st.caption("Help us improve the model (Assignment Task 2)")

with st.form("feedback_form"):
    rating = st.slider("How accurate/relevant was this prediction?", 1, 5, 3)
    user_comment = st.text_area("Any suggestions for improvement?")
    user_role = st.selectbox("Your Role", ["Student", "Instructor", "Peer", "Other"])
    
    fb_submit = st.form_submit_button("Submit Feedback")

if fb_submit:
    # Save feedback to CSV
    feedback_file = "suggestions_dataset.csv"
    new_feedback = pd.DataFrame({
        'Role': [user_role],
        'Rating': [rating],
        'Comment': [user_comment],
        'Date': [pd.Timestamp.now()]
    })
    
    if os.path.exists(feedback_file):
        new_feedback.to_csv(feedback_file, mode='a', header=False, index=False)
    else:
        new_feedback.to_csv(feedback_file, index=False)
        
    st.success("Thank you! Your feedback has been recorded.")

# Footer
st.markdown("---")
st.caption("Deployment by Assignment 04 System | Streamlit")
