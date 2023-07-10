import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components


st.set_page_config(
    page_title= "Profiling de Dados",
    page_icon= "📊",
    layout= "wide",
    initial_sidebar_state= "collapsed",
    menu_items= {
        'About': "Desenvolvido por:\n- Dayvson Moura;\n- José Roberto;\n- Thales Mayrinck;\n- Vinícius Gustavo."
    }
)


@st.cache_data
def gerar_relatorio():
   df = pd.read_csv("./data/telco_churn_data.csv")
   return ProfileReport(df, title="Análise Exploratória").to_html()


def header():
    st.header("Análise Exploratória de Dados")
    st.markdown("""
                Esta página apresenta gráficos e relatórios a partir do <i>dataset</i>.
                """,
                unsafe_allow_html= True)


def profile():
    report = gerar_relatorio()
    components.html(report, height=1000, width=1120, scrolling=True)


def build_graphics():
    st.header("Gráficos")


header()
profile()
build_graphics()
