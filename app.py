# Customer Segmentation App
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from elbow import get_optimum_k

#page config
st.set_page_config(page_title="Customer Segmentation App", page_icon="./assets/logo.png", layout="wide")

with st.sidebar:
    st.title("Customer Segmentation App")
    st.image("./assets/logo.png")


file = st.file_uploader("Upload your CSV file", type=["csv", "xlsx"])
if file:
    if file.name.split(".")[-1] == "csv":
        df = pd.read_csv(file, index_col=0)
    else:
        df = pd.read_excel(file, index_col=0)

    columns = st.multiselect("Select Features", options=df.columns )
    df = df[columns]
    scaled = StandardScaler()
    scaled.fit_transform(df)



    #kMeans clustering
    optimum_k = get_optimum_k(df)
    model = KMeans(n_clusters=optimum_k)
    model.fit(df)
    df["labels"] = model.labels_

    #scatter chart
    st.scatter_chart(df, x="Annual Income (k$)", y="Spending Score (1-100)", color="labels")  


