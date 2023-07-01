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

st.header("Gráficos")

opcoes_de_graficos = [ 
    "Percentual de clientes que saíram do banco e permaneceram",
    "Percentual de homens e mulheres", 
    "Percentual por idade",
    "Percentual por score de satisfação", 
    "Reclamou x Saiu",
    ]
grafico = st.selectbox("Selecione um gráfico:", opcoes_de_graficos, 0)

if grafico == "Percentual de clientes que saíram do banco e permaneceram":
    contagem_torta_saiu = df['Exited'].value_counts()
    fig, ax = plt.subplots(figsize=(8,6))
    ax.pie(
        contagem_torta_saiu,
        labels=['Não saiu', 'Saiu'],
        pctdistance=0.4,
        autopct='%1.2f%%', 
        radius=0.6,
        )
    ax.legend(['Não saiu', 'Saiu'])
    ax.set_ylabel('Status')
    st.pyplot(fig)

elif grafico == "Percentual de homens e mulheres":
    total = df["Gender"].value_counts()
    fig, ax = plt.subplots(figsize=(8,6))
    ax.pie(
        total, 
        labels=['Homens', 'Mulheres'], 
        pctdistance=0.4, 
        autopct='%1.2f%%', 
        radius=0.6,
        )
    ax.legend(['Homens', "Mulheres"])
    st.pyplot(fig)

elif grafico == "Percentual por idade":
    total_idades = df['Age'].value_counts(normalize=True) * 100
    fig, ax = plt.subplots()
    ax.bar(total_idades.index, total_idades.values)
    ax.set_xlabel('Idades')
    ax.set_ylabel('Porcentagem')
    st.pyplot(fig)

elif grafico == "Percentual por score de satisfação":
    contagem_barras_satisfacao = df['Satisfaction Score'].value_counts(normalize=True) * 100
    fig, ax = plt.subplots(figsize=(8,6))
    contagem_barras_satisfacao.plot.barh()
    ax.set_xlabel('Porcentagem')
    ax.set_ylabel("Score de satisfação")
    ax.set_xlim (0, 25)

    #adiciona os valores à direita das barras
    for i, valor in enumerate(contagem_barras_satisfacao):
        ax.annotate(
            f' {valor:.2f}%', 
            (valor, i), 
            va='center',
            ) 

    st.pyplot(fig)

elif grafico == "Reclamou x Saiu":
    #criando pequenos dataframes
    reclamou_saiu = df[(df['Complain'] == 1) & (df['Exited'] == 1)].shape[0]
    reclamou_nao_saiu = df[(df['Complain'] == 1) & (df['Exited'] == 0)].shape[0]
    nao_reclamou_saiu = df[(df['Complain'] == 0) & (df['Exited'] == 1)].shape[0]
    nao_reclamou_nao_saiu = df[(df['Complain'] == 0) & (df['Exited'] == 0)].shape[0]

    #unindo eles num dicionário
    contagem_barras_grupos = {'Status': ['Reclamou e saiu.', 'Reclamou e não saiu.', 'Não reclamou e saiu.', 'Não reclamou e não saiu.'],
            'Count': [reclamou_saiu, reclamou_nao_saiu, nao_reclamou_saiu, nao_reclamou_nao_saiu]}

    #criando um dataframe do dicionário
    df_grupos = pd.DataFrame(contagem_barras_grupos)
    df_grupos = df_grupos.sort_values (by='Count')

    fig, ax = plt.subplots(figsize=(8, 6))

    #dizendo os nomes e os valores das colunas
    ax.bar(x=df_grupos['Status'], height=df_grupos['Count'])

    #escrevendo os valores em cima das colunas
    for i, v in enumerate(df_grupos['Count']):
        ax.text(i, v, str(v), ha='center', va='bottom')

    st.pyplot(fig)