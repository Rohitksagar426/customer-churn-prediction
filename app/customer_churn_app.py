import streamlit as st
import pandas as pd
import joblib 
import plotly.graph_objects as go


st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.markdown("""
    <style>
    .stButton>button {
        width: 100%
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 10px;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
    }
    .churn { background-color: #ffebee; color: #c62828; }
    .no-churn { background-color: #e8f5e9; color: #2e7d32; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        # model = joblib.load("customer_churn_model.pkl")
        pipeline = joblib.load("customer_churn_pipeline.pkl")
        return pipeline
    except FileNotFoundError:
        st.error("Model files not found. Run the notebook first.")
        return None
    
pipeline = load_model()

st.title("Customer Churn Prediction")
st.markdown("Predict customer churn with Machine Learning")
st.markdown("---")

with st.sidebar:
        st.header("Model Info")
        st.info("""
        **XGBoost classifier**
        - Test Accuracy: ~77%
        - Training : 7,043 customers
        - Features: 19 attributes
        - Balanced with SMOTE
        """)
        
        st.header("Top Predictors")
        st.markdown("""
        - Contract type
        - Tenure duration
        - Monthly charges
        - Internet service
        - Payment method
        """)
        
        st.markdown("---")
        st.caption("Build with Streamlit & Scikit-learn")



if pipeline is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        
        st.subheader("Services")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No","Yes","No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    
    with col2:
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        
        st.subheader("Account")
        tenure = st.slider("Tenure (Months)", 0, 72, 12 )
        contract = st.selectbox("Contract", ["Month-to-Month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method",
                                      ["Electronic check", "Mailed check",
                                       "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 50.0, 5.0)
        total_charges = st.number_input("Total charges ($)", 0.0, 10000.0, monthly_charges * tenure, 50.0)
        
    st.markdown("---")
    
    if st.button("Predict Churn"):
        input_data = {
            'gender': gender, 
            'SeniorCitizen': senior_citizen, 
            'Partner': partner,
            'Dependents' : dependents, 
            'tenure':tenure, 
            'PhoneService': phone_service,
            'MultipleLines' : multiple_lines, 
            'InternetService' : internet_service,
            'OnlineSecurity': online_security, 
            'OnlineBackup' : online_backup,
            'DeviceProtection' : device_protection, 
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv, 
            'StreamingMovies': streaming_movies,
            'Contract' : contract, 
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method, 
            'MonthlyCharges': monthly_charges,
            'TotalCharges' : total_charges
        }
        
        input_df = pd.DataFrame([input_data])
        
        # Encode Binary Features
        # Make prediction (ONLY ONCE)
        prediction = pipeline.predict(input_df)[0]
        proba = pipeline.predict_proba(input_df)[0]

        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        # Column 1 → Prediction
        with col1:
            if prediction == 1:
                st.markdown('<div class="prediction-box churn">WILL CHURN</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="prediction-box no-churn">WILL STAY</div>', unsafe_allow_html=True)
                
        # Column 2 → Confidence
        with col2:
            confidence = proba[prediction]
            st.metric("Confidence", f"{confidence * 100:.1f}%")
        
        # Column 3 - Risk
        with col3:
            churn_probability = proba[1]

            if churn_probability > 0.7:
                risk = "High"
            elif churn_probability > 0.4:
                risk = "Medium"
            else:
                risk = "Low"

            st.metric("Risk Level", risk)
        
        fig = go.Figure(data = [
            go.Bar(name = "No Churn", x= ['Probability'], y= [proba[0]], marker_color = '#2ecc71'),
            go.Bar(name = 'Churn', x = ['Probability'], y = [proba[1]], marker_color = '#e74c3c')
        ])
        fig.update_layout(
            title = "Prediction Probabilities",
            yaxis_title = "Probability",
            barmode = 'group',
            height = 350
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if prediction == 1:
            st.warning("**At Risk:** Offer retention incentives, upgrade to longer contract, provide better support")
        else:
            st.success("**Low Risk:** Continue excellent service, send surveys, offer loyalty benefits")
        
    
    
else:
    st.error("Model not found. Run the notebook to generate model files.")
        