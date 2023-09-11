import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


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

data = pd.read_csv("./data/telco_churn_data.csv")

def dispersao():
    scatter_fig = px.scatter(data, x=selected_column_x, y=selected_column_y, title='Gráfico de Dispersão')
    st.plotly_chart(scatter_fig)

def histograma():
    hist_fig = px.histogram(data, x=selected_column_x, title='Histograma')
    st.plotly_chart(hist_fig)

def grafico_barras():
    bar_fig = px.bar(data, x=selected_column_x, y=selected_column_y, title='Gráfico de Barras')
    st.plotly_chart(bar_fig)

def grafico_pizza():
    pie_fig = px.pie(data, names=selected_column_x, title='Gráfico de Pizza')
    st.plotly_chart(pie_fig)

def dispersao_3d():
    scatter_3d_fig = go.Figure(data=[go.Scatter3d(
    x=data[selected_column_x],
    y=data[selected_column_y],
    z=data[selected_column_Z],
    mode='markers',
    marker=dict(size=5, colorscale='Viridis'),
    text=data.index,
    name='Gráfico de Dispersão 3D'
)])
    scatter_3d_fig.update_layout(scene=dict(zaxis_title='Coluna Z'))
    st.plotly_chart(scatter_3d_fig)

header()
filters_section()
selected_column_x = st.selectbox("Selecione a coluna X", data.columns)
selected_column_y = st.selectbox("Selecione a coluna Y", data.columns)
selected_column_Z = st.selectbox("Selecione a coluna Z", data.columns)
dispersao()
histograma()
grafico_barras()
grafico_pizza()
dispersao_3d() 