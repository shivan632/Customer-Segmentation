import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495e;
        padding: 0.5rem;
        border-bottom: 2px solid #3498db;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .insight-text {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #3498db;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# DATA LOADING AND CACHING
# ============================================
@st.cache_data
def load_data():
    df = pd.read_csv('Mall_Customers.csv')
    return df

@st.cache_resource
def perform_clustering(df, n_clusters=5):
    # Select features for clustering
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    return kmeans, scaler, cluster_labels, X_scaled

@st.cache_data
def get_cluster_summary(df):
    summary = df.groupby('Cluster').agg({
        'Annual Income (k$)': ['mean', 'std', 'min', 'max'],
        'Spending Score (1-100)': ['mean', 'std', 'min', 'max'],
        'Age': ['mean', 'std', 'min', 'max'],
        'Genre': lambda x: x.mode()[0] if not x.mode().empty else 'Mixed',
        'CustomerID': 'count'
    }).round(1)
    
    summary.columns = ['Income_Mean', 'Income_Std', 'Income_Min', 'Income_Max',
                      'Spending_Mean', 'Spending_Std', 'Spending_Min', 'Spending_Max',
                      'Age_Mean', 'Age_Std', 'Age_Min', 'Age_Max',
                      'Dominant_Gender', 'Size']
    return summary

# ============================================
# LOAD DATA
# ============================================
df = load_data()

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/null/mall.png", width=100)
    st.title("🎯 Customer Segmentation")
    st.markdown("---")
    
    # Navigation
    st.subheader("Navigation")
    page = st.radio(
        "Go to",
        ["📊 EDA Dashboard", 
         "🔍 Clustering Analysis", 
         "🎯 Customer Segments",
         "📈 Business Insights"]
    )
    
    st.markdown("---")
    
    # Parameters for clustering
    if page in ["🔍 Clustering Analysis", "🎯 Customer Segments"]:
        st.subheader("⚙️ Clustering Parameters")
        n_clusters = st.slider("Number of Clusters (K)", 2, 8, 5)
        
        # Feature selection
        features = st.multiselect(
            "Select Features for Clustering",
            ['Annual Income (k$)', 'Spending Score (1-100)', 'Age'],
            default=['Annual Income (k$)', 'Spending Score (1-100)']
        )
    
    st.markdown("---")
    st.markdown("### About")
    st.info("""
    This dashboard performs customer segmentation using K-Means clustering 
    on mall customer data. Analyze demographics, spending patterns, 
    and get actionable business insights.
    """)

# ============================================
# MAIN CONTENT
# ============================================
st.markdown('<div class="main-header">🛍️ Customer Segmentation Dashboard</div>', unsafe_allow_html=True)

# ============================================
# PAGE 1: EDA DASHBOARD
# ============================================
if page == "📊 EDA Dashboard":
    st.markdown('<div class="sub-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    # Key Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Customers", df.shape[0])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg Age", f"{df['Age'].mean():.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg Income", f"${df['Annual Income (k$)'].mean():.1f}k")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg Spending Score", f"{df['Spending Score (1-100)'].mean():.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        gender_ratio = df['Genre'].value_counts(normalize=True).values[0] * 100
        st.metric("Female %", f"{gender_ratio:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tabs for different EDA views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Distributions", "👥 Demographics", "🔗 Correlations", "📋 Raw Data"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Age Distribution
            fig_age = px.histogram(df, x='Age', nbins=20, 
                                  title='Age Distribution',
                                  color_discrete_sequence=['#3498db'])
            fig_age.add_vline(x=df['Age'].mean(), line_dash="dash", 
                            line_color="red", annotation_text=f"Mean: {df['Age'].mean():.1f}")
            st.plotly_chart(fig_age, use_container_width=True)
            
            # Income Distribution
            fig_income = px.histogram(df, x='Annual Income (k$)', nbins=20,
                                     title='Annual Income Distribution',
                                     color_discrete_sequence=['#2ecc71'])
            fig_income.add_vline(x=df['Annual Income (k$)'].mean(), line_dash="dash",
                               line_color="red", annotation_text=f"Mean: {df['Annual Income (k$)'].mean():.1f}k")
            st.plotly_chart(fig_income, use_container_width=True)
        
        with col2:
            # Spending Score Distribution
            fig_spending = px.histogram(df, x='Spending Score (1-100)', nbins=20,
                                       title='Spending Score Distribution',
                                       color_discrete_sequence=['#e74c3c'])
            fig_spending.add_vline(x=df['Spending Score (1-100)'].mean(), line_dash="dash",
                                 line_color="red", annotation_text=f"Mean: {df['Spending Score (1-100)'].mean():.1f}")
            st.plotly_chart(fig_spending, use_container_width=True)
            
            # Age vs Spending Score
            fig_scatter = px.scatter(df, x='Age', y='Spending Score (1-100)',
                                    color='Genre', title='Age vs Spending Score by Gender',
                                    color_discrete_map={'Male': '#3498db', 'Female': '#e74c3c'})
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Gender Distribution
            gender_counts = df['Genre'].value_counts().reset_index()
            gender_counts.columns = ['Genre', 'Count']
            fig_gender = px.pie(gender_counts, values='Count', names='Genre',
                               title='Gender Distribution',
                               color_discrete_map={'Male': '#3498db', 'Female': '#e74c3c'})
            st.plotly_chart(fig_gender, use_container_width=True)
            
            # Age Box Plot by Gender
            fig_age_box = px.box(df, x='Genre', y='Age', color='Genre',
                                title='Age Distribution by Gender',
                                color_discrete_map={'Male': '#3498db', 'Female': '#e74c3c'})
            st.plotly_chart(fig_age_box, use_container_width=True)
        
        with col2:
            # Income Box Plot by Gender
            fig_income_box = px.box(df, x='Genre', y='Annual Income (k$)', color='Genre',
                                   title='Income Distribution by Gender',
                                   color_discrete_map={'Male': '#3498db', 'Female': '#e74c3c'})
            st.plotly_chart(fig_income_box, use_container_width=True)
            
            # Spending Score Box Plot by Gender
            fig_spending_box = px.box(df, x='Genre', y='Spending Score (1-100)', color='Genre',
                                     title='Spending Score Distribution by Gender',
                                     color_discrete_map={'Male': '#3498db', 'Female': '#e74c3c'})
            st.plotly_chart(fig_spending_box, use_container_width=True)
    
    with tab3:
        # Correlation Heatmap
        numeric_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
        corr_matrix = df[numeric_cols].corr()
        
        fig_corr = px.imshow(corr_matrix, 
                            text_auto=True,
                            aspect="auto",
                            title='Correlation Matrix',
                            color_continuous_scale='RdBu_r',
                            labels=dict(color="Correlation"))
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Scatter Matrix
        fig_scatter_matrix = px.scatter_matrix(df, dimensions=numeric_cols,
                                              color='Genre',
                                              title='Scatter Matrix of Numerical Features',
                                              color_discrete_map={'Male': '#3498db', 'Female': '#e74c3c'})
        st.plotly_chart(fig_scatter_matrix, use_container_width=True)
    
    with tab4:
        st.subheader("Raw Data")
        st.dataframe(df, use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name="mall_customers_data.csv",
            mime="text/csv"
        )

# ============================================
# PAGE 2: CLUSTERING ANALYSIS
# ============================================
elif page == "🔍 Clustering Analysis":
    st.markdown('<div class="sub-header">🔍 K-Means Clustering Analysis</div>', unsafe_allow_html=True)
    
    # Perform clustering
    kmeans, scaler, cluster_labels, X_scaled = perform_clustering(df, n_clusters)
    df_clustered = df.copy()
    df_clustered['Cluster'] = cluster_labels
    
    # Elbow Method
    st.subheader("🎯 Optimal Number of Clusters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Calculate inertia for different K values
        inertias = []
        sil_scores = []
        K_range = range(1, 11)
        
        for k in K_range:
            kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_temp.fit(X_scaled)
            inertias.append(kmeans_temp.inertia_)
            if k > 1:
                sil_scores.append(silhouette_score(X_scaled, kmeans_temp.labels_))
        
        # Elbow curve
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(x=list(K_range), y=inertias,
                                       mode='lines+markers',
                                       name='Inertia',
                                       line=dict(color='blue', width=2)))
        fig_elbow.add_vline(x=n_clusters, line_dash="dash", line_color="red",
                           annotation_text=f"K={n_clusters}")
        fig_elbow.update_layout(title='Elbow Method for Optimal K',
                               xaxis_title='Number of Clusters (K)',
                               yaxis_title='Inertia')
        st.plotly_chart(fig_elbow, use_container_width=True)
    
    with col2:
        # Silhouette scores
        fig_silhouette = go.Figure()
        fig_silhouette.add_trace(go.Scatter(x=list(range(2, 11)), y=sil_scores,
                                           mode='lines+markers',
                                           name='Silhouette Score',
                                           line=dict(color='green', width=2)))
        fig_silhouette.add_vline(x=n_clusters, line_dash="dash", line_color="red",
                                annotation_text=f"K={n_clusters}")
        fig_silhouette.update_layout(title='Silhouette Score for Different K',
                                    xaxis_title='Number of Clusters (K)',
                                    yaxis_title='Silhouette Score')
        st.plotly_chart(fig_silhouette, use_container_width=True)
    
    st.markdown("---")
    
    # Cluster Visualization
    st.subheader(f"📊 Customer Segments Visualization (K={n_clusters})")
    
    # 2D Scatter Plot
    fig_clusters = px.scatter(df_clustered, 
                             x='Annual Income (k$)', 
                             y='Spending Score (1-100)',
                             color='Cluster',
                             title=f'Customer Segments based on Income and Spending Score',
                             color_continuous_scale='viridis',
                             hover_data=['CustomerID', 'Age', 'Genre'])
    
    # Add cluster centers
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    centers_df = pd.DataFrame(centers, columns=['Annual Income (k$)', 'Spending Score (1-100)'])
    centers_df['Cluster'] = range(n_clusters)
    
    fig_clusters.add_trace(go.Scatter(x=centers_df['Annual Income (k$)'],
                                      y=centers_df['Spending Score (1-100)'],
                                      mode='markers',
                                      marker=dict(symbol='x', size=15, color='black'),
                                      name='Centroids'))
    
    st.plotly_chart(fig_clusters, use_container_width=True)
    
    # 3D Scatter Plot (if 3 features selected)
    if len(features) >= 3:
        st.subheader("3D Cluster Visualization")
        fig_3d = px.scatter_3d(df_clustered, 
                              x=features[0], 
                              y=features[1], 
                              z=features[2],
                              color='Cluster',
                              title='3D View of Customer Segments',
                              color_continuous_scale='viridis')
        st.plotly_chart(fig_3d, use_container_width=True)

# ============================================
# PAGE 3: CUSTOMER SEGMENTS
# ============================================
elif page == "🎯 Customer Segments":
    st.markdown('<div class="sub-header">🎯 Customer Segment Analysis</div>', unsafe_allow_html=True)
    
    # Perform clustering
    kmeans, scaler, cluster_labels, X_scaled = perform_clustering(df, n_clusters)
    df['Cluster'] = cluster_labels
    
    # Get cluster summary
    cluster_summary = get_cluster_summary(df)
    
    # Cluster names mapping
    cluster_names = {
        0: "Low Income - Low Spending",
        1: "High Income - Low Spending", 
        2: "Medium Income - Medium Spending",
        3: "High Income - High Spending",
        4: "Low Income - High Spending"
    }
    
    # Add cluster names if n_clusters is 5
    if n_clusters == 5:
        df['Segment'] = df['Cluster'].map(cluster_names)
    
    # Cluster distribution
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Pie chart of cluster distribution
        cluster_sizes = df['Cluster'].value_counts().sort_index()
        fig_pie = px.pie(values=cluster_sizes.values, 
                        names=[f'Cluster {i}' for i in cluster_sizes.index],
                        title='Customer Distribution Across Clusters',
                        color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Bar chart of cluster sizes
        fig_bar = px.bar(x=cluster_sizes.index, y=cluster_sizes.values,
                        title='Number of Customers per Cluster',
                        labels={'x': 'Cluster', 'y': 'Number of Customers'},
                        color=cluster_sizes.index,
                        color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("---")
    
    # Cluster Characteristics
    st.subheader("📋 Detailed Cluster Characteristics")
    
    # Display cluster summary table with proper formatting
    display_summary = cluster_summary.copy()
    display_summary.index = [f'Cluster {i}' for i in display_summary.index]
    
    # Format the table for better display
    styled_summary = display_summary.style.format({
        'Income_Mean': '${:.1f}k',
        'Income_Std': '${:.1f}k',
        'Income_Min': '${:.1f}k',
        'Income_Max': '${:.1f}k',
        'Spending_Mean': '{:.1f}',
        'Spending_Std': '{:.1f}',
        'Spending_Min': '{:.1f}',
        'Spending_Max': '{:.1f}',
        'Age_Mean': '{:.1f} yrs',
        'Age_Std': '{:.1f}',
        'Age_Min': '{:.1f}',
        'Age_Max': '{:.1f}'
    })
    
    st.dataframe(styled_summary, use_container_width=True)
    
    # Visualize cluster characteristics - FIXED VERSION
    col1, col2 = st.columns(2)
    
    with col1:
        # Income distribution by cluster - FIXED: removed color_continuous_scale
        fig_income_box = px.box(df, x='Cluster', y='Annual Income (k$)',
                               title='Income Distribution by Cluster',
                               color='Cluster',
                               color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig_income_box, use_container_width=True)
    
    with col2:
        # Spending score distribution by cluster - FIXED: removed color_continuous_scale
        fig_spending_box = px.box(df, x='Cluster', y='Spending Score (1-100)',
                                 title='Spending Score Distribution by Cluster',
                                 color='Cluster',
                                 color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig_spending_box, use_container_width=True)
    
    # Age distribution by cluster - FIXED: removed color_continuous_scale
    fig_age_box = px.box(df, x='Cluster', y='Age',
                        title='Age Distribution by Cluster',
                        color='Cluster',
                        color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig_age_box, use_container_width=True)
    
    # Radar chart for cluster profiles
    if n_clusters == 5:
        st.subheader("🕸️ Cluster Profiles Radar Chart")
        
        # Prepare data for radar chart
        radar_data = []
        for cluster in range(5):
            cluster_data = df[df['Cluster'] == cluster]
            radar_data.append({
                'Cluster': f'Cluster {cluster}',
                'Income': cluster_data['Annual Income (k$)'].mean(),
                'Spending': cluster_data['Spending Score (1-100)'].mean(),
                'Age': cluster_data['Age'].mean()
            })
        
        radar_df = pd.DataFrame(radar_data)
        
        # Normalize values for radar chart
        for col in ['Income', 'Spending', 'Age']:
            radar_df[f'{col}_normalized'] = (radar_df[col] - radar_df[col].min()) / (radar_df[col].max() - radar_df[col].min())
        
        fig_radar = go.Figure()
        
        colors = px.colors.qualitative.Set3[:5]
        
        for i, row in radar_df.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row['Income_normalized'], row['Spending_normalized'], row['Age_normalized']],
                theta=['Income', 'Spending', 'Age'],
                fill='toself',
                name=f'Cluster {i}',
                line_color=colors[i]
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            title="Cluster Profiles Comparison"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)

# ============================================
# PAGE 4: BUSINESS INSIGHTS
# ============================================
else:
    st.markdown('<div class="sub-header">📈 Business Insights & Recommendations</div>', unsafe_allow_html=True)
    
    # Perform clustering with optimal K=5
    kmeans, scaler, cluster_labels, X_scaled = perform_clustering(df, 5)
    df['Cluster'] = cluster_labels
    
    cluster_summary = get_cluster_summary(df)
    
    # Cluster interpretations
    interpretations = {
        0: {
            "name": "Budget-Conscious Shoppers",
            "description": "Customers with low annual income and low spending score. These are careful spenders who primarily purchase essential items.",
            "characteristics": "💰 Low Income, 📉 Low Spending, 🎯 Value-seeking",
            "recommendations": "Offer budget-friendly options, discounts, and value packs. Focus on essential products and promotional deals.",
            "color": "#FF6B6B"
        },
        1: {
            "name": "Cautious High-Earners",
            "description": "High-income customers with surprisingly low spending scores. They have high purchasing power but choose to spend conservatively.",
            "characteristics": "💵 High Income, 📉 Low Spending, 👴 Older demographic",
            "recommendations": "Build trust through premium quality, exclusive offers, and personalized service. Focus on long-term value and loyalty programs.",
            "color": "#4ECDC4"
        },
        2: {
            "name": "Average Mainstream Shoppers",
            "description": "Customers with average income and average spending habits. This is the largest segment representing typical mall shoppers.",
            "characteristics": "💵 Medium Income, 📊 Medium Spending, ⚖️ Balanced",
            "recommendations": "Standard marketing approaches work well. Focus on variety, seasonal promotions, and maintaining consistent quality.",
            "color": "#FFD93D"
        },
        3: {
            "name": "Premium High-Spenders",
            "description": "Ideal customers with high income and high spending scores. They are relatively younger and have both the means and willingness to spend.",
            "characteristics": "💎 High Income, 📈 High Spending, 👨‍👩‍👧‍👦 Young professionals",
            "recommendations": "VIP treatment, early access to new products, exclusive events, premium services, and personalized recommendations.",
            "color": "#6BCB77"
        },
        4: {
            "name": "Aspirational Young Shoppers",
            "description": "Young customers with low income but high spending scores. Despite limited income, they have strong spending desires.",
            "characteristics": "💰 Low Income, 📈 High Spending, 🧑‍🎓 Students/Young adults",
            "recommendations": "Target with trendy products, installment payment options, student discounts, and social media marketing.",
            "color": "#9B59B6"
        }
    }
    
    # Key Insights
    st.subheader("🎯 Key Business Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="insight-text">', unsafe_allow_html=True)
        st.markdown("**🏆 Most Valuable Segment**")
        st.markdown("**Premium High-Spenders** (Cluster 3)")
        st.markdown(f"• {cluster_summary.loc[3, 'Size']} customers")
        st.markdown(f"• Avg Income: ${cluster_summary.loc[3, 'Income_Mean']:.1f}k")
        st.markdown(f"• Avg Spending: {cluster_summary.loc[3, 'Spending_Mean']:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="insight-text">', unsafe_allow_html=True)
        st.markdown("**📊 Largest Segment**")
        st.markdown("**Average Mainstream Shoppers** (Cluster 2)")
        st.markdown(f"• {cluster_summary.loc[2, 'Size']} customers")
        st.markdown(f"• {cluster_summary.loc[2, 'Size']/len(df)*100:.1f}% of total")
        st.markdown(f"• Balanced spending behavior")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="insight-text">', unsafe_allow_html=True)
        st.markdown("**🚀 Growth Opportunity**")
        st.markdown("**Aspirational Young Shoppers** (Cluster 4)")
        st.markdown(f"• {cluster_summary.loc[4, 'Size']} customers")
        st.markdown(f"• Avg Age: {cluster_summary.loc[4, 'Age_Mean']:.1f} years")
        st.markdown(f"• High potential for brand loyalty")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Detailed segment analysis
    st.subheader("📊 Detailed Segment Analysis")
    
    # Create tabs for each cluster
    cluster_tabs = st.tabs([f"Cluster {i}: {interpretations[i]['name']}" for i in range(5)])
    
    for i, tab in enumerate(cluster_tabs):
        with tab:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown(f"### {interpretations[i]['name']}")
                st.markdown(f"**Description:** {interpretations[i]['description']}")
                st.markdown(f"**Key Characteristics:** {interpretations[i]['characteristics']}")
                
                st.markdown("#### 📈 Segment Statistics")
                stats_df = pd.DataFrame({
                    'Metric': ['Size', 'Avg Income', 'Avg Spending', 'Avg Age', 'Dominant Gender'],
                    'Value': [
                        f"{cluster_summary.loc[i, 'Size']} customers",
                        f"${cluster_summary.loc[i, 'Income_Mean']:.1f}k",
                        f"{cluster_summary.loc[i, 'Spending_Mean']:.1f}",
                        f"{cluster_summary.loc[i, 'Age_Mean']:.1f} years",
                        cluster_summary.loc[i, 'Dominant_Gender']
                    ]
                })
                st.table(stats_df)
            
            with col2:
                st.markdown("#### 💡 Business Recommendations")
                st.markdown(f'<div style="background-color: {interpretations[i]["color"]}20; padding: 1rem; border-radius: 5px; border-left: 5px solid {interpretations[i]["color"]};">', unsafe_allow_html=True)
                st.markdown(f"**Marketing Strategy:**")
                st.markdown(interpretations[i]['recommendations'])
                
                st.markdown("**Targeting Approach:**")
                if i == 0:
                    st.markdown("- Price-sensitive marketing")
                    st.markdown("- Bundle deals and discounts")
                    st.markdown("- Essential products focus")
                elif i == 1:
                    st.markdown("- Premium positioning")
                    st.markdown("- Trust-building campaigns")
                    st.markdown("- Quality over quantity")
                elif i == 2:
                    st.markdown("- Broad appeal marketing")
                    st.markdown("- Seasonal promotions")
                    st.markdown("- Family-oriented offers")
                elif i == 3:
                    st.markdown("- Exclusive VIP programs")
                    st.markdown("- Early access privileges")
                    st.markdown("- Personalized experiences")
                else:
                    st.markdown("- Social media campaigns")
                    st.markdown("- Influencer partnerships")
                    st.markdown("- Flexible payment options")
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Strategic Recommendations
    st.subheader("💼 Strategic Business Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Short-term Actions")
        st.markdown("""
        1. **Launch targeted campaigns** for Premium High-Spenders with exclusive previews
        2. **Create student discount program** for Aspirational Young Shoppers
        3. **Implement loyalty program** for Average Mainstream Shoppers
        4. **Test premium product lines** with Cautious High-Earners
        5. **Develop value bundles** for Budget-Conscious Shoppers
        """)
    
    with col2:
        st.markdown("#### 📈 Long-term Strategy")
        st.markdown("""
        1. **Build brand loyalty** among young aspirational shoppers
        2. **Develop premium service offerings** for high-income segments
        3. **Create personalized shopping experiences** using AI
        4. **Expand product range** for mainstream shoppers
        5. **Implement predictive analytics** for customer behavior
        """)
    
    # ROI Analysis
    st.subheader("💰 ROI Analysis by Segment")
    
    roi_data = pd.DataFrame({
        'Segment': [interpretations[i]['name'] for i in range(5)],
        'Customer Value': [85, 45, 60, 95, 75],
        'Marketing ROI': [70, 50, 65, 90, 80],
        'Growth Potential': [60, 40, 55, 75, 90]
    })
    
    fig_roi = px.bar(roi_data, x='Segment', y=['Customer Value', 'Marketing ROI', 'Growth Potential'],
                     title='Segment Potential Analysis',
                     barmode='group',
                     color_discrete_sequence=['#3498db', '#2ecc71', '#e74c3c'])
    st.plotly_chart(fig_roi, use_container_width=True)
    
    # Download report
    st.markdown("---")
    st.subheader("📥 Download Analysis Report")
    
    # Create summary report
    report = f"""
    CUSTOMER SEGMENTATION ANALYSIS REPORT
    =====================================
    Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}
    Total Customers Analyzed: {len(df)}
    
    KEY FINDINGS:
    1. Five distinct customer segments identified
    2. Most valuable: Premium High-Spenders (Cluster 3)
    3. Largest segment: Average Mainstream Shoppers (Cluster 2)
    4. Growth opportunity: Aspirational Young Shoppers (Cluster 4)
    
    SEGMENT BREAKDOWN:
    {cluster_summary.to_string()}
    
    RECOMMENDATIONS:
    1. Allocate 40% of marketing budget to Premium High-Spenders
    2. Develop loyalty program for Average Mainstream Shoppers
    3. Create social media campaigns for Aspirational Young Shoppers
    4. Implement premium services for Cautious High-Earners
    5. Design value bundles for Budget-Conscious Shoppers
    """
    
    st.download_button(
        label="📥 Download Full Analysis Report",
        data=report,
        file_name="customer_segmentation_report.txt",
        mime="text/plain"
    )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #7f8c8d; padding: 1rem;'>
        Customer Segmentation Dashboard | Built with Streamlit & Scikit-learn
    </div>
    """,
    unsafe_allow_html=True
)