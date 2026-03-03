import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# --- Page Config ---
st.set_page_config(page_title="AI Customer Segments", layout="wide")

st.title("🎯 Customer Personality Analysis & Recommendation System")
st.markdown("""
This app uses **Unsupervised Machine Learning (K-Means)** to segment customers 
and provide tailored marketing recommendations.
""")

# --- Sidebar ---
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload your Cleaned_Customer_Data.csv",
    type=["csv"]
)

# Allow user to choose number of clusters
k = st.sidebar.slider("Select Number of Clusters", 2, 8, 4)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # --- Feature Selection ---
    features = [
        'Age', 'Income', 'Total_Spent', 'Enrollment_Days', 'Children',
        'NumWebPurchases', 'NumCatalogPurchases',
        'NumStorePurchases', 'NumWebVisitsMonth'
    ]

    # Check if all features exist
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns in dataset: {missing_cols}")
        st.stop()

    # Handle Missing Values
    if df[features].isnull().sum().sum() > 0:
        st.warning("Missing values detected. Filling with median values.")
        df[features] = df[features].fillna(df[features].median())

    X = df[features]

    # --- Scaling ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- KMeans Clustering (on full scaled data) ---
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # --- Silhouette Score ---
    score = silhouette_score(X_scaled, df['Cluster'])
    st.sidebar.metric("Silhouette Score", f"{score:.3f}")

    # --- PCA for Visualization Only ---
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    df["PCA1"] = X_pca[:, 0]
    df["PCA2"] = X_pca[:, 1]

    # --- Dashboard Layout ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Customer Segments Visualization (PCA)")
        fig = px.scatter(
            df,
            x="PCA1",
            y="PCA2",
            color=df['Cluster'].astype(str),
            hover_data=['Age', 'Income', 'Total_Spent'],
            labels={'PCA1': 'Spending Power', 'PCA2': 'Life Stage'},
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Cluster Distribution")
        cluster_counts = df['Cluster'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'Count']

        fig_pie = px.pie(
            cluster_counts,
            values='Count',
            names='Cluster',
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()
    cluster_profile = df.groupby("Cluster")[features].mean().round(1)
    # --- Recommendation Engine ---
    st.subheader("🔍 Get Marketing Recommendation")

    if "ID" not in df.columns:
        st.error("Dataset must contain an 'ID' column.")
        st.stop()

    selected_id = st.selectbox("Select a Customer ID", df['ID'].unique())

    if st.button("Generate Strategy"):
        customer = df[df['ID'] == selected_id].iloc[0]
        cluster = customer['Cluster']

        # Dynamic strategy dictionary (based on cluster count)
        strategies = {
            0: {"Persona": "Value Shopper", 
                "Strategy": "Offer discount coupons and loyalty rewards.",
                "Top": "Essential Goods"},
            1: {"Persona": "Premium Spender",
                "Strategy": "VIP membership, exclusive previews, premium bundles.",
                "Top": "Luxury Products"},
            2: {"Persona": "Digital Buyer",
                "Strategy": "Mobile push notifications & flash sales.",
                "Top": "Online Deals"},
            3: {"Persona": "Family Planner",
                "Strategy": "Bulk discounts and family combo packs.",
                "Top": "Groceries & Household"}
        }

        res = strategies.get(cluster, {
            "Persona": "General Customer",
            "Strategy": "Standard marketing campaigns.",
            "Top": "Mixed Products"
        })

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Persona", res["Persona"])
        c2.metric("Income", f"${customer['Income']:,.0f}")
        c3.metric("Total Spent", f"${customer['Total_Spent']:,.0f}")

        st.success(f"**Marketing Action:** {res['Strategy']}")
        st.info(f"**Target Products:** {res['Top']}")

else:
    st.info("Please upload the 'Cleaned_Customer_Data.csv' to begin analysis.")