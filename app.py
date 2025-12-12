
import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Real Estate Investment Advisor", layout="wide")
st.title("Real Estate Investment Advisor")

DATA_PATH = r"C:\Users\Shreya Ghosal\Downloads\india_housing_prices.csv"
df = pd.read_csv(DATA_PATH)

# quick cleaning and feature creation (mirrors eda logic)
if 'Price_in_Lakhs' in df.columns and 'Size_in_SqFt' in df.columns:
    df['Price_per_SqFt'] = (df['Price_in_Lakhs'] * 1e5) / df['Size_in_SqFt']
else:
    df['Price_per_SqFt'] = np.nan

# expected growth heuristic
base_5yr = (1 + 0.08)**5 - 1
if 'City' in df.columns and df['Price_per_SqFt'].notna().sum()>0:
    city_med = df.groupby('City')['Price_per_SqFt'].median()
else:
    city_med = pd.Series()

# Sidebar: inputs
st.sidebar.header("Enter property details")
city = st.sidebar.selectbox("City", options=['Unknown'] + sorted(df['City'].dropna().unique().tolist()) if 'City' in df.columns else ['Unknown'])
property_type = st.sidebar.selectbox("Property Type", options=['Unknown'] + sorted(df['Property_Type'].dropna().unique().tolist()) if 'Property_Type' in df.columns else ['Unknown'])
size = st.sidebar.number_input("Size (SqFt)", min_value=100, max_value=20000, value=1000)
price_lakhs = st.sidebar.number_input("Current Price (Lakhs)", min_value=0.01, max_value=100000.0, value=50.0, step=0.1)
nearby_schools = st.sidebar.number_input("Nearby Schools (count/rating)", min_value=0, max_value=100, value=3)
nearby_hospitals = st.sidebar.number_input("Nearby Hospitals (count)", min_value=0, max_value=50, value=2)
bhk = st.sidebar.selectbox("BHK", options=[1,2,3,4,5])

if st.sidebar.button("Assess Investment"):
    # compute price_per_sqft for this property
    price_per_sqft = (price_lakhs * 1e5) / max(size,1)
    # compute expected growth using heuristic
    if city in city_med.index:
        city_median_val = city_med.loc[city]
    else:
        city_median_val = df['Price_per_SqFt'].median() if df['Price_per_SqFt'].notna().sum()>0 else price_per_sqft
    rel_price = price_per_sqft / (city_median_val + 1e-9)
    expected_5yr_growth = base_5yr * (1 + (1 - rel_price)*0.5)
    expected_5yr_growth = float(np.clip(expected_5yr_growth, -0.5, 2.0))

    # classification rule
    is_good = expected_5yr_growth > 0.20
    confidence = min(max(abs( (expected_5yr_growth - 0.2) / 0.2 ), 0.05), 0.99)  # heuristic confidence

    st.subheader("Investment Recommendation")
    if is_good:
        st.success(f"Good Investment ✅   (Confidence: {confidence:.2%})")
    else:
        st.error(f"Not a Good Investment ❌   (Confidence: {confidence:.2%})")

    st.subheader("Estimated Price after 5 years (heuristic)")
    est_price_5yr = price_lakhs * (1 + expected_5yr_growth)
    st.write(f"Current price: **{price_lakhs:.2f} Lakhs**")
    st.write(f"Estimated price after 5 years: **{est_price_5yr:.2f} Lakhs** (growth: {expected_5yr_growth:.2%})")

    # optionally use pre-trained models if present
    use_models = False
    if os.path.exists('rf_classifier.joblib') and os.path.exists('rf_regressor.joblib') and os.path.exists('model_features.joblib'):
        try:
            clf = joblib.load('rf_classifier.joblib')
            reg = joblib.load('rf_regressor.joblib')
            model_feats = joblib.load('model_features.joblib')
            # build feature vector using get_dummies like in training
            row = {
                'Size_in_SqFt': size,
                'Price_per_SqFt': price_per_sqft,
                'Nearby_Schools': nearby_schools,
                'Nearby_Hospitals': nearby_hospitals,
                'Age_of_Property': df['Age_of_Property'].median() if 'Age_of_Property' in df.columns else 10
            }
            row['City'] = city
            row['Property_Type'] = property_type
            x_new = pd.DataFrame([row])
            x_new_d = pd.get_dummies(x_new, drop_first=True)


        # ... use models ...
    except Exception as e:
        st.warning("Model files present but failed to load: " + str(e))
else:
    st.info("Model backend unavailable — using heuristic rules only. (Install joblib in requirements.txt for full model behavior.)")
            # ensure all model_features present
            for c in model_feats:
                if c not in x_new_d.columns:
                    x_new_d[c] = 0
            x_new_d = x_new_d[model_feats]
            pred_label = clf.predict(x_new_d)[0]
            pred_proba = clf.predict_proba(x_new_d).max()
            pred_price_model = reg.predict(x_new_d)[0]
            st.caption("Model-based predictions (pre-trained models detected)")
            if pred_label==1:
                st.success(f"Model: Good Investment ✅ (prob {pred_proba:.2%})")
            else:
                st.error(f"Model: Not a Good Investment ❌ (prob {pred_proba:.2%})")
            st.write(f"Model predicted current price (Lakhs): **{pred_price_model:.2f}**")
            # feature importance
            try:
                imp = clf.feature_importances_
                feat_imp = pd.DataFrame({'feature': model_feats, 'importance': imp}).sort_values('importance', ascending=False).head(10)
                st.subheader("Top feature importances (classifier)")
                st.table(feat_imp.set_index('feature'))
            except Exception as e:
                st.write("Feature importance not available:", e)
            use_models = True
        except Exception as e:
            st.write("Failed to load or use pre-trained models:", e)

    # Visual insights
    st.subheader("Visual insights (dataset)")
    # City median price-per-sqft bar
    if 'City' in df.columns and df['Price_per_SqFt'].notna().sum()>0:
        city_med_df = df.groupby('City')['Price_per_SqFt'].median().sort_values(ascending=False).reset_index()
        st.caption("Top cities by median price per sqft")
        st.bar_chart(city_med_df.set_index('City')['Price_per_SqFt'].head(20))
    else:
        st.write("No city-price_per_sqft info available.")

    # Distribution of price per sqft
    if df['Price_per_SqFt'].notna().sum()>0:
        fig, ax = plt.subplots(figsize=(6,3))
        sns.histplot(df['Price_per_SqFt'].replace([np.inf,-np.inf], np.nan).dropna(), bins=60, ax=ax)
        ax.set_title("Price per SqFt distribution")
        st.pyplot(fig)
    else:
        st.write("No Price_per_SqFt distribution available.")

    # Top localities
    if 'Locality' in df.columns and 'Price_in_Lakhs' in df.columns:
        top_loc = df.groupby('Locality')['Price_in_Lakhs'].median().sort_values(ascending=False).head(10)
        st.subheader("Top 10 Localities by Median Price")
        st.table(top_loc.reset_index().rename(columns={'Price_in_Lakhs':'Median_Price_Lakhs'}))
    else:
        st.write("No Locality price info available.")

    # Extra suggestion
    st.markdown("**Note:** This app uses heuristic rules by default to give instant results. For model-backed predictions, run a training script to produce `rf_classifier.joblib`, `rf_regressor.joblib`, and `model_features.joblib` then restart the app.")


