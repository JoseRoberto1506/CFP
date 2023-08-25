import pandas as pd
import streamlit as st
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import plotly.express as px


clustering_cols = ['Gender','Age','Married','Dependents']

st.set_page_config(
    page_title= "Clusterização",
    page_icon= "🧠",
    layout= "wide",
    initial_sidebar_state= "collapsed",
    menu_items= {
        'About': "Desenvolvido por:\n- Dayvson Moura;\n- José Roberto;\n- Thales Mayrinck;\n- Vinícius Gustavo."
    }
)


def main():
    header()
    leitura = preprocessamento()
    clusterizado=clusterizacao(leitura,StandardScaler())



def header():
    st.header("Machine Learning")
    st.markdown("""
                Esta página apresenta os modelos de <i>machine learning</i> criados e os seus resultados.            
                """,
                unsafe_allow_html= True)


def preprocessamento():
    st.markdown("### Pré-processamento")
    st.markdown("""
                Nesta etapa, 
                """,
                unsafe_allow_html=True)
    df = ler_dataset()
    if st.checkbox("Mostrar dataset após o pré-processamento dos dados"):
        st.dataframe(df)
    st.divider()
    
    return df



@st.cache_data
def ler_dataset():
    df = pd.read_csv("./data/telco_churn_data.csv")
    return df.copy()


def transformacao(dfrecebido):
    df=dfrecebido.copy()
    st.markdown("### Transformação")
    st.markdown("""
                Nesta etapa, 
                """,
                unsafe_allow_html=True)
    
    colunas_categoricas_binarias = ['Married','Dependents','Under 30','Senior Citizen']
    df['Gender'].replace({'Male': 0, 'Female': 1}, inplace=True)
    for i, coluna in enumerate(colunas_categoricas_binarias):
        df[coluna].replace({'No': 0, 'Yes': 1}, inplace=True)

    
    
    if st.checkbox("Mostrar dataset após a transformação dos dados"):
        st.dataframe(df)
    st.divider()

    return df






def clusterizacao(dfrecebido,scaler:TransformerMixin=None):
    st.markdown("### Clusterização")
    df_final_de_verdade=dfrecebido.copy()
    df_clusterizando_final=transformacao(df_final_de_verdade)
    df_clusterizando=df_clusterizando_final.copy()
    if scaler is not None:
        df_clusterizando = scale(df_clusterizando, scaler)
    modelo = KMeans(n_clusters=11, n_init=10, random_state=42)
    dfcluster=df_clusterizando[clustering_cols]
    
    dfcluster['Cluster']=modelo.fit_predict(dfcluster)
    df_final_de_verdade['Cluster']=dfcluster['Cluster']
    st.dataframe(df_final_de_verdade)

    return df_final_de_verdade



def scale(dfrecebido:pd.DataFrame, scaler:TransformerMixin):
    df=dfrecebido.copy()
    scaling_cols = [x for x in ['Age','Number of Dependents'] if x in clustering_cols]
    for c in scaling_cols:
        vals = df[[c]].values
        df[c] = scaler.fit_transform(vals)
    return df
    




main()