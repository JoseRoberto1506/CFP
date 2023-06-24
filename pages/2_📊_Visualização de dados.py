import streamlit as st
import pandas as pd

st.set_page_config(
    page_title= "Visualização de dados",
    page_icon= "📊",
    layout= "wide",
    initial_sidebar_state= "auto",
    menu_items= {
        'About': "Desenvolvido por:\n- Dayvson Moura;\n- José Roberto;\n- Thales Mayrinck;\n- Vinícius Gustavo."
    }
)


@st.cache_data
def ler_dataset():
    return pd.read_csv("./data/Customer-Churn-Records.csv")


df = ler_dataset()

st.header("Visualização de dados")
st.markdown("""
            Esta página apresenta alguns gráficos e opções de filtragem a partir do <i>dataset</i>.
            """,
            unsafe_allow_html= True)

st.markdown("#### Filtros")
filtros_padrao = df.columns.to_list()
filtros = st.multiselect('Filtrar por:', df.columns, filtros_padrao[:10])
ordernar_por = st.multiselect('Ordenar por:', filtros, filtros[0])
st.dataframe(df.filter(filtros).sort_values(ordernar_por))

# st.markdown("#### Gráficos")
