import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

# Page configuration
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a3a5c;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4a6a8c;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a3a5c;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6a8aac;
    }
    .status-pass {
        color: #27ae60;
        font-weight: 600;
    }
    .status-fail {
        color: #e74c3c;
        font-weight: 600;
    }
    .status-warn {
        color: #f39c12;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING FUNCTIONS
# ============================================================

@st.cache_resource
def load_data():
    """Load all data and models"""
    try:
        X = joblib.load('data/processed/X_processed.joblib')
        y = joblib.load('data/processed/y.joblib')
        feature_names = joblib.load('data/processed/feature_names.joblib')
        df_raw = pd.read_csv('data/raw/bank_churn_modelling.csv')
        model = joblib.load('models/best_model.joblib')
        preprocessor = joblib.load('models/preprocessor.joblib')
        
        # Try to load fair model
        try:
            fair_model = joblib.load('models/fair_model.joblib')
        except:
            fair_model = None
        
        return X, y, feature_names, df_raw, model, preprocessor, fair_model
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None, None, None

@st.cache_data
def get_predictions(_model, X):
    """Get predictions and probabilities"""
    y_pred = _model.predict(X)
    y_proba = _model.predict_proba(X)[:, 1]
    return y_pred, y_proba

@st.cache_data
def calculate_fairness_metrics(y_true, y_pred, sensitive_features):
    """Calculate fairness metrics"""
    dp = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
    eo = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features)
    return dp, eo

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.image("https://img.icons8.com/color/96/000000/bank.png", width=80)
st.sidebar.title("🏦 Churn Analytics")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["📊 Overview", "🔮 Predictions", "⚖️ Fairness", "📖 Explainability"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Version:** v1.0.0")
st.sidebar.markdown("**Last Updated:** August 2026")

# Load data
X, y, feature_names, df_raw, model, preprocessor, fair_model = load_data()

if X is None:
    st.error("⚠️ Failed to load data. Please check your file paths.")
    st.stop()

# ============================================================
# PAGE: OVERVIEW
# ============================================================

if page == "📊 Overview":
    st.markdown('<p class="main-header">📊 Model Overview</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Churn Prediction Model Performance Summary</p>', unsafe_allow_html=True)
    
    # Get predictions
    y_pred, y_proba = get_predictions(model, X)
    
    # Metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{accuracy_score(y, y_pred):.1%}</div>
            <div class="metric-label">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{roc_auc_score(y, y_proba):.3f}</div>
            <div class="metric-label">ROC-AUC</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{recall_score(y, y_pred):.1%}</div>
            <div class="metric-label">Recall</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{precision_score(y, y_pred):.1%}</div>
            <div class="metric-label">Precision</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{f1_score(y, y_pred):.3f}</div>
            <div class="metric-label">F1-Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Two columns: Confusion Matrix and ROC
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y, y_pred)
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=['Not Churn', 'Churn'],
            y=['Not Churn', 'Churn'],
            color_continuous_scale='Blues',
            title="Confusion Matrix"
        )
        fig_cm.update_layout(height=400)
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col2:
        st.subheader("ROC Curve")
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y, y_proba)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC (AUC = {roc_auc_score(y, y_proba):.3f})',
            line=dict(color='#1a3a5c', width=3)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(color='red', dash='dash')
        ))
        fig_roc.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=400
        )
        st.plotly_chart(fig_roc, use_container_width=True)
    
    # Feature Importance
    st.markdown("---")
    st.subheader("Top 10 Feature Importance (SHAP)")
    
    # Load SHAP values if available, otherwise use feature importance
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        shap_importance = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(shap_importance)[-10:][::-1]
        top_features = [feature_names[i] for i in top_idx]
        top_importance = shap_importance[top_idx]
    except:
        # Fallback to feature importance
        top_idx = np.argsort(model.feature_importances_)[-10:][::-1]
        top_features = [feature_names[i] for i in top_idx]
        top_importance = model.feature_importances_[top_idx]
    
    fig_imp = px.bar(
        x=top_importance,
        y=top_features,
        orientation='h',
        title="Feature Importance",
        labels={'x': 'Importance', 'y': 'Feature'},
        color=top_importance,
        color_continuous_scale='Blues'
    )
    fig_imp.update_layout(height=400)
    st.plotly_chart(fig_imp, use_container_width=True)

# ============================================================
# PAGE: PREDICTIONS
# ============================================================

elif page == "🔮 Predictions":
    st.markdown('<p class="main-header">🔮 Make Predictions</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enter customer details to get churn prediction</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Information")
        
        # Input fields
        credit_score = st.slider("Credit Score", 300, 850, 650)
        geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 100, 35)
        tenure = st.slider("Tenure (years)", 0, 10, 5)
        balance = st.number_input("Balance (€)", min_value=0, max_value=300000, value=100000)
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
        has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
        is_active = st.selectbox("Is Active Member", ["Yes", "No"])
        estimated_salary = st.number_input("Estimated Salary (€)", min_value=0, max_value=300000, value=100000)
    
    with col2:
        st.subheader("Prediction Result")
        
        if st.button("🔮 Predict Churn Risk", use_container_width=True):
            # Prepare input
            input_data = pd.DataFrame([{
                'CreditScore': credit_score,
                'Geography': geography,
                'Gender': gender,
                'Age': age,
                'Tenure': tenure,
                'Balance': balance,
                'NumOfProducts': num_products,
                'HasCrCard': 1 if has_cr_card == "Yes" else 0,
                'IsActiveMember': 1 if is_active == "Yes" else 0,
                'EstimatedSalary': estimated_salary
            }])
            
            # Preprocess
            X_input = preprocessor.transform(input_data)
            
            # Predict
            proba = model.predict_proba(X_input)[0][1]
            pred = model.predict(X_input)[0]
            
            # Display result
            risk_level = "High" if proba > 0.7 else "Medium" if proba > 0.3 else "Low"
            color = "#e74c3c" if proba > 0.7 else "#f39c12" if proba > 0.3 else "#27ae60"
            
            st.markdown(f"""
            <div style="background-color: #f8f9fa; border-radius: 10px; padding: 2rem; text-align: center;">
                <div style="font-size: 3rem; font-weight: 700; color: {color};">
                    {proba:.1%}
                </div>
                <div style="font-size: 1.2rem; color: #4a6a8c;">
                    Churn Probability
                </div>
                <div style="font-size: 1.5rem; margin-top: 1rem; color: {color};">
                    Risk Level: {risk_level}
                </div>
                <div style="font-size: 1rem; margin-top: 0.5rem; color: #4a6a8c;">
                    Prediction: {pred}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show SHAP explanation
            st.subheader("What Drives This Prediction?")
            try:
                import shap
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_input)
                
                # Create SHAP bar chart
                shap_vals = shap_values[0]
                top_indices = np.argsort(np.abs(shap_vals))[-5:][::-1]
                
                for idx in top_indices:
                    feature = feature_names[idx]
                    value = shap_vals[idx]
                    direction = "🔴 Increases risk" if value > 0 else "🟢 Decreases risk"
                    st.write(f"**{feature}**: {direction} ({abs(value):.3f})")
            except:
                st.info("SHAP explanation not available for this prediction")

# ============================================================
# PAGE: FAIRNESS
# ============================================================

elif page == "⚖️ Fairness":
    st.markdown('<p class="main-header">⚖️ Fairness Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Model fairness across protected attributes</p>', unsafe_allow_html=True)
    
    # Get predictions
    y_pred, y_proba = get_predictions(model, X)
    
    # Prepare fairness data
    fairness_df = pd.DataFrame({
        'Geography': df_raw['Geography'][:len(y_pred)],
        'Gender': df_raw['Gender'][:len(y_pred)],
        'Age': df_raw['Age'][:len(y_pred)],
        'y_true': y,
        'y_pred': y_pred
    })
    
    fairness_df['AgeGroup'] = pd.cut(
        fairness_df['Age'],
        bins=[0, 18, 30, 40, 50, 60, 100],
        labels=['Under 18', '18-29', '30-39', '40-49', '50-59', '60+']
    )
    
    # Fairness metrics
    col1, col2, col3 = st.columns(3)
    
    # Geography
    dp_geo, eo_geo = calculate_fairness_metrics(
        fairness_df['y_true'], 
        fairness_df['y_pred'], 
        fairness_df['Geography']
    )
    
    with col1:
        st.subheader("Geography")
        st.metric("Demographic Parity", f"{dp_geo:.4f}", delta=None)
        st.metric("Equalized Odds", f"{eo_geo:.4f}", delta=None)
        status = "✅ Pass" if dp_geo < 0.10 else "❌ Fail"
        st.markdown(f"**Status:** {status}")
    
    # Gender
    dp_gender, eo_gender = calculate_fairness_metrics(
        fairness_df['y_true'], 
        fairness_df['y_pred'], 
        fairness_df['Gender']
    )
    
    with col2:
        st.subheader("Gender")
        st.metric("Demographic Parity", f"{dp_gender:.4f}", delta=None)
        st.metric("Equalized Odds", f"{eo_gender:.4f}", delta=None)
        status = "✅ Pass" if dp_gender < 0.10 else "❌ Fail"
        st.markdown(f"**Status:** {status}")
    
    # Age
    dp_age, eo_age = calculate_fairness_metrics(
        fairness_df['y_true'], 
        fairness_df['y_pred'], 
        fairness_df['AgeGroup']
    )
    
    with col3:
        st.subheader("Age")
        st.metric("Demographic Parity", f"{dp_age:.4f}", delta=None)
        st.metric("Equalized Odds", f"{eo_age:.4f}", delta=None)
        status = "✅ Pass" if dp_age < 0.10 else "❌ Fail"
        st.markdown(f"**Status:** {status}")
    
    st.markdown("---")
    
    # Selection rates by group
    st.subheader("Selection Rates by Protected Attribute")
    
    fig = make_subplots(rows=1, cols=3, subplot_titles=["Geography", "Gender", "Age Group"])
    
    # Geography
    geo_rates = fairness_df.groupby('Geography')['y_pred'].mean().reset_index()
    fig.add_trace(
        go.Bar(x=geo_rates['Geography'], y=geo_rates['y_pred'], name='Geography'),
        row=1, col=1
    )
    fig.add_hline(y=fairness_df['y_pred'].mean(), line_dash="dash", line_color="red", row=1, col=1)
    
    # Gender
    gender_rates = fairness_df.groupby('Gender')['y_pred'].mean().reset_index()
    fig.add_trace(
        go.Bar(x=gender_rates['Gender'], y=gender_rates['y_pred'], name='Gender'),
        row=1, col=2
    )
    fig.add_hline(y=fairness_df['y_pred'].mean(), line_dash="dash", line_color="red", row=1, col=2)
    
    # Age
    age_rates = fairness_df.groupby('AgeGroup')['y_pred'].mean().reset_index()
    fig.add_trace(
        go.Bar(x=age_rates['AgeGroup'].astype(str), y=age_rates['y_pred'], name='Age Group'),
        row=1, col=3
    )
    fig.add_hline(y=fairness_df['y_pred'].mean(), line_dash="dash", line_color="red", row=1, col=3)
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Fairness summary table
    st.subheader("Fairness Summary")
    
    summary_data = {
        'Protected Attribute': ['Geography', 'Gender', 'Age'],
        'Demographic Parity': [dp_geo, dp_gender, dp_age],
        'Equalized Odds': [eo_geo, eo_gender, eo_age],
        'Threshold (DP < 0.10)': [
            '✅ Pass' if dp_geo < 0.10 else '❌ Fail',
            '✅ Pass' if dp_gender < 0.10 else '❌ Fail',
            '✅ Pass' if dp_age < 0.10 else '❌ Fail'
        ]
    }
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

# ============================================================
# PAGE: EXPLAINABILITY
# ============================================================

else:
    st.markdown('<p class="main-header">📖 Explainability</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Understand why the model makes its predictions</p>', unsafe_allow_html=True)
    
    # Select customer
    st.subheader("Select a Customer to Explain")
    
    customer_id = st.selectbox(
        "Choose a customer",
        options=df_raw['CustomerId'].head(100).tolist(),
        format_func=lambda x: f"Customer {x} (Index {df_raw[df_raw['CustomerId'] == x].index[0]})"
    )
    
    if customer_id:
        idx = df_raw[df_raw['CustomerId'] == customer_id].index[0]
        
        # Get prediction
        X_input = X[idx:idx+1]
        proba = model.predict_proba(X_input)[0][1]
        pred = model.predict(X_input)[0]
        
        # Display customer info
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Customer Details:**")
            st.write(f"- Age: {df_raw.iloc[idx]['Age']}")
            st.write(f"- Geography: {df_raw.iloc[idx]['Geography']}")
            st.write(f"- Gender: {df_raw.iloc[idx]['Gender']}")
            st.write(f"- Tenure: {df_raw.iloc[idx]['Tenure']} years")
            st.write(f"- Balance: €{df_raw.iloc[idx]['Balance']:,.2f}")
            st.write(f"- Products: {df_raw.iloc[idx]['NumOfProducts']}")
        
        with col2:
            st.write("**Prediction:**")
            risk_color = "#e74c3c" if proba > 0.7 else "#f39c12" if proba > 0.3 else "#27ae60"
            st.markdown(f"""
            <div style="background-color: #f8f9fa; border-radius: 10px; padding: 1.5rem; text-align: center;">
                <div style="font-size: 2.5rem; font-weight: 700; color: {risk_color};">
                    {proba:.1%}
                </div>
                <div style="color: #4a6a8c;">Churn Probability</div>
                <div style="margin-top: 0.5rem; color: {risk_color}; font-weight: 600;">
                    {'⚠️ High Risk' if proba > 0.7 else '📊 Medium Risk' if proba > 0.3 else '✅ Low Risk'}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # SHAP explanation
        st.markdown("---")
        st.subheader("What Drives This Prediction? (SHAP)")
        
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_input)
            
            # SHAP bar chart
            shap_vals = shap_values[0]
            top_indices = np.argsort(np.abs(shap_vals))[-10:][::-1]
            
            shap_df = pd.DataFrame({
                'Feature': [feature_names[i] for i in top_indices],
                'SHAP Value': [shap_vals[i] for i in top_indices],
                'Impact': ['🔴 Increases Risk' if shap_vals[i] > 0 else '🟢 Decreases Risk' for i in top_indices]
            })
            
            fig_shap = px.bar(
                shap_df,
                x='SHAP Value',
                y='Feature',
                orientation='h',
                color='Impact',
                color_discrete_map={'🔴 Increases Risk': '#e74c3c', '🟢 Decreases Risk': '#27ae60'},
                title="Feature Contributions to Prediction"
            )
            fig_shap.update_layout(height=400)
            st.plotly_chart(fig_shap, use_container_width=True)
            
        except Exception as e:
            st.info(f"SHAP explanation not available: {e}")