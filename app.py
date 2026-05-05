import streamlit as st
import numpy as np
import joblib
import pandas as pd

st.set_page_config(page_title="Telecom Churn Intelligence", layout="wide")
st.markdown('<link rel="icon" href="data:,">', unsafe_allow_html=True)

# --- LOADING MODELS, SCALER, AND DATABASE ---
model = joblib.load('churn_model.pkl')
kmeans = joblib.load('segmentation_model.pkl')
scaler = joblib.load('scaler.pkl')
lookup_db = pd.read_csv('customer_lookup_database.csv', dtype={'Phone_Number': str})

all_columns = list(model.get_booster().feature_names)

tenure_col = [col for col in all_columns if 'tenure' in col.lower()][0]
charge_col = [col for col in all_columns if 'charge' in col.lower()][0]

st.title("Strategic Churn & Segment Predictor")

# --- SIDEBAR: SEARCH BY PHONE ---
st.sidebar.header("🔍 Customer Lookup")
search_phone = st.sidebar.text_input("Enter Phone Number", placeholder="e.g. 08012345678")

if st.button("Search and Analyze"):
    if search_phone:
        clean_search = str(search_phone).strip()
        customer_record = lookup_db[lookup_db['Phone_Number'].astype(str).str.strip() == clean_search].copy()
        
        if not customer_record.empty:
            st.sidebar.success("✅ Customer Record Found!")

            st.subheader("📋 Customer Profile (Key Details)")
            
            display_df = customer_record.copy()
            if 'gender' in display_df.columns:
                display_df['gender'] = display_df['gender'].replace({1: 'Male', 0: 'Female'})

            display_cols = ['Cluster', tenure_col, charge_col, 'gender', 'Contract_Month-to-month', 
                            'InternetService_Fiber optic', 'OnlineSecurity_No', 'TechSupport_No', 
                            'StreamingMovies_Yes', 'Partner', 'Dependents']
            
            existing_display_cols = [c for c in display_cols if c in display_df.columns]
            st.dataframe(display_df[existing_display_cols], use_container_width=True)
            st.divider()

            input_df = customer_record[all_columns].copy()
            if 'gender' in input_df.columns:
                input_df['gender'] = input_df['gender'].replace({'Male': 1, 'Female': 0}).astype(int)

            prob = model.predict_proba(input_df)[0][1]
            cluster = int(customer_record['Cluster'].values[0])

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Churn Risk Assessment")
                st.metric("Probability", f"{prob*100:.2f}%")
                if prob > 0.5:
                    st.error("⚠️ VERDICT: HIGH RISK")
                else:
                    st.success("✅ VERDICT: STABLE")

            with col2:
                st.subheader("Customer Segmentation")
                st.info(f"Assigned Group: Cluster {cluster}")
                
                if cluster == 1:
                    st.write("**Profile:** At-Risk Starter")
                    st.write("**Strategy:** Offer Annual Contract discounts.")
                elif cluster == 2:
                    st.write("**Profile:** Loyal Power User")
                    st.write("**Strategy:** Enroll in Premium Loyalty.")
                else:
                    st.write("**Profile:** Basic Traditionalist")
                    st.write("**Strategy:** Upsell Digital Bundles.")
                    
            st.divider()
            if prob > 0.5:
                st.warning(f"**Executive Summary:** This customer has a {prob*100:.1f}% churn risk. Strategic intervention is advised.")
        else:
            st.error("❌ PHONE NUMBER NOT FOUND")
    else:
        st.warning("Please enter a phone number in the sidebar.")