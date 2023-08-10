import streamlit as st
import pandas as pd


st.set_page_config(
    page_title= "Filtragem de dados",
    page_icon= "📊",
    layout= "wide",
    initial_sidebar_state= "collapsed",
    menu_items= {
        'About': "Desenvolvido por:\n- Dayvson Moura;\n- José Roberto;\n- Thales Mayrinck;\n- Vinícius Gustavo."
    }
)


@st.cache_data
def ler_dataset():
    return pd.read_csv("./data/telco_churn_data.csv")


def header():
    st.header("Visualização de dados")
    st.markdown("""
                Esta página apresenta opções de filtragem e ordenação do <i>dataset</i>, permitindo uma visualização de dados mais dinâmica.
                """,
                unsafe_allow_html= True)


def filters_section():
    st.markdown("#### Filtros e Ordenação")
    df = ler_dataset()
    filtros_padrao = df.columns.to_list()
    filtros = st.multiselect(
        'Filtrar por:', 
        df.columns, 
        filtros_padrao[1:10]
        )
    ordernar_por = st.multiselect(
        'Ordenar por:', 
        filtros, 
        filtros[0]
        )
    st.dataframe(df.filter(filtros).sort_values(ordernar_por))


header()
filters_section()
