# 🛍️ Customer Segmentation using K-Means Clustering

> Internship Project – Data Analysis & Machine Learning  
> Author: Prakhar Srivastava

---

## 📌 Project Overview

Customer segmentation is a strategic business technique used to divide customers into meaningful groups based on shared characteristics.

This internship project focuses on analyzing customer data and applying K-Means clustering to identify distinct customer segments based on:

- Annual Income
- Spending Score

The objective is to help businesses better understand customer behavior and enable data-driven marketing decisions.

---

## 🎯 Business Objective

A retail company aims to:

- Understand customer purchasing behavior
- Identify high-value and potential customers
- Create targeted marketing campaigns
- Improve customer retention
- Maximize profitability

By segmenting customers, the company can design personalized strategies for each group.

---

## 📂 Dataset Description

The dataset includes the following attributes:

| Column | Description |
|--------|------------|
| CustomerID | Unique identifier for each customer |
| Gender | Male / Female |
| Age | Age of the customer |
| Annual Income (k$) | Annual income in thousand dollars |
| Spending Score (1–100) | Score assigned based on purchasing behavior |

---

## 🛠️ Tools & Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- K-Means Clustering
- Elbow Method
- Jupyter Notebook

---

## 🔍 Exploratory Data Analysis (EDA)

EDA was conducted to:

- Check for missing or inconsistent data
- Understand feature distributions
- Analyze relationships between income and spending
- Identify potential clustering patterns

### Key Observations:

- Spending behavior is not directly proportional to income.
- Customers naturally form distinct groups when visualized.
- Some high-income customers exhibit low spending behavior, indicating potential growth opportunities.

---

## 📊 Optimal Cluster Selection – Elbow Method

The Elbow Method was used to determine the optimal number of clusters.

It evaluates clustering performance by measuring Within-Cluster Sum of Squares (WCSS) and identifying the point where adding more clusters provides diminishing improvement.

### Result:
The optimal number of clusters identified was **5**.

---

## 🤖 Customer Segmentation using K-Means

Using Annual Income and Spending Score, customers were grouped into five distinct clusters.

### Identified Customer Segments:

| Cluster | Segment Description |
|----------|--------------------|
| Cluster 1 | High Income – High Spending |
| Cluster 2 | High Income – Low Spending |
| Cluster 3 | Low Income – High Spending |
| Cluster 4 | Low Income – Low Spending |
| Cluster 5 | Average Income – Average Spending |

---

## 🧠 Business Insights & Strategic Recommendations

### 🔵 High Income – High Spending
- Premium customers
- Focus on loyalty programs and exclusive memberships
- Offer early access to new products

### 🔴 High Income – Low Spending
- Untapped potential segment
- Use personalized marketing campaigns
- Provide targeted promotions

### 🟢 Low Income – High Spending
- Deal-driven customers
- Offer seasonal discounts and bundles

### 🟡 Low Income – Low Spending
- Budget-focused segment
- Promote affordable product categories

### 🟣 Average Customers
- Stable contributors to revenue
- Maintain regular engagement strategies

---

## 💼 Business Impact

This segmentation approach helps the company to:

- Improve marketing ROI
- Optimize customer targeting
- Increase customer lifetime value (CLV)
- Reduce marketing waste
- Enable strategic decision-making

---

## 📁 Project Structure

Customer-Segmentation  
│  
├── Customer_Segmentation.ipynb  
├── Mall_Customers.csv  
├── README.md  

---

## 🚀 How to Use This Project

1. Clone the repository from GitHub.  
2. Install the required Python libraries.  
3. Open the Jupyter Notebook file.  
4. Run all cells to reproduce analysis and clustering results.

---

## 📊 Future Enhancements

- Include Age-based segmentation
- Implement Hierarchical Clustering
- Apply DBSCAN for density-based clustering
- Build an interactive dashboard (Power BI / Tableau)
- Deploy as a Streamlit web application

---

## 📌 Conclusion

This internship project successfully demonstrates:

- End-to-end data analysis workflow
- Customer behavior exploration
- Optimal cluster selection using the Elbow Method
- Customer segmentation using K-Means
- Translation of machine learning results into business insights

The final model segmented customers into five meaningful groups, providing actionable strategies for targeted marketing and business growth.

---

## 👨‍💻 Author

Shivan Mishra
Data Scientist Intern  
GitHub: https://github.com/shivan632