import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components
import pages.utils.graficos_aed as ga


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
def profile(df):
   report = ProfileReport(df, title="Análise Exploratória").to_html()
   components.html(report, height=1000, width=1120, scrolling=True)


def main():
    header()
    df = pd.read_csv("./data/telco_churn_data.csv")
    build_graphics(df)
    profile(df)



def header():
    st.header("Análise Exploratória de Dados")
    st.markdown("""
                Esta página apresenta a análise exploratória de dados realizada a partir do <i>dataset</i>.
                """,
                unsafe_allow_html= True)


def build_graphics(df):
    st.header("Gráficos relativos ao churn")
    df_filtrado_por_churn_positivo = df[df['Churn Value'] == 1]
    mapa_de_opcoes = {'Idade': 'Age',
                      'Indicações feitas': 'Number of Referrals',
                      'Número de dependentes': 'Number of Dependents',
                      'Satisfação': 'Customer Satisfaction',
                      'Tempo na empresa': 'Tenure in Months',
                      }
    grafico_selecionado = st.selectbox("Selecione o gráfico:",
                                         ["Idade",
                                          "Indicações feitas",
                                          "Método pagamento",
                                          "Motivo do churn x Satisfação",
                                          "Número de dependentes",
                                          "Satisfação",
                                          "Solicitações do serviço de cliente",
                                          "Tempo na empresa",
                                          "Tipo de contrato",
                                          ])
    
    if grafico_selecionado == "Método pagamento":
        ga.metodo_pagamento(df_filtrado_por_churn_positivo)
    elif grafico_selecionado == "Motivo do churn x Satisfação":
        ga.motivo_churn_x_satisfacao(df)
    elif grafico_selecionado == "Solicitações do serviço de cliente":
        ga.solicitacoes_serv_cliente(df)
    elif grafico_selecionado == "Tipo de contrato":
        ga.tipo_contrato(df_filtrado_por_churn_positivo)
    else:
        ga.grafico_em_barra(df_filtrado_por_churn_positivo, mapa_de_opcoes[grafico_selecionado])
    
main()
