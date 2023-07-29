import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


st.set_page_config(
    page_title= "Agrupamento de Dados",
    page_icon= "",
    layout= "wide",
    initial_sidebar_state= "collapsed",
    menu_items= {
        'About': "Desenvolvido por:\n- Dayvson Moura;\n- José Roberto;\n- Thales Mayrinck;\n- Vinícius Gustavo."
    }
)

def header():
    st.header("Visualização de Teste")
    st.markdown("""
                Esta página apresenta algumas opções de filtragem a partir do <i>dataset</i>.
                """,
                unsafe_allow_html= True)

ds = pd.read_csv("./data/telco_churn_data.csv")

#Limpeza dos Dados

ds = ds.drop('Customer ID', axis = 1)

filtro = ds['Churn Value'] == 0
linhas_a_excluir = ds[filtro].sample(n=3305, random_state=42)
df_filtrado = ds.drop(linhas_a_excluir.index) 


#Transformação dos Dados

df_filtrado2 = df_filtrado
df_filtrado2['Offer'] = df_filtrado2['Offer'].replace({'Offer A': 1, 'Offer B': 2, 'Offer C': 3, 'Offer D': 4, 'Offer E': 5})
df_filtrado3 = df_filtrado2
df_filtrado3['Offer'] = df_filtrado3['Offer'].fillna(0)