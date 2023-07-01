import streamlit as st
import pandas as pd
import pandas_profiling as pp
import numpy as np
import streamlit.components.v1 as components
import matplotlib.pyplot as plt


st.set_page_config(
    page_title= "Análise Exploratória de Dados",
    page_icon= "📊",
    layout= "wide",
    initial_sidebar_state= "auto",
    menu_items= {
        'About': "Desenvolvido por:\n- Dayvson Moura;\n- José Roberto;\n- Thales Mayrinck;\n- Vinícius Gustavo."
    }
)


st.header("Análise Exploratória de Dados")
st.markdown("""
            Esta página apresenta gráficos e relatórios desta ferramenta a partir do <i>dataset</i>.
            """,
            unsafe_allow_html= True)

@st.cache_data
def gerar_relatorio():
   return pp.ProfileReport(df, title="Análise Exploratória").to_html()

df = pd.read_csv("./data/Customer-Churn-Records.csv")

report = gerar_relatorio()
components.html(report, height=1000, width=1120,scrolling=True)