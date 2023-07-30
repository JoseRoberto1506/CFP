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
dsmutavel = ds.copy()

#Limpeza dos Dados

dsmutavel = dsmutavel.drop('Customer ID', axis = 1)

filtro = dsmutavel['Churn Value'] == 0
linhas_a_excluir = dsmutavel[filtro].sample(n=3305, random_state=42)
dsmutavel = dsmutavel.drop(linhas_a_excluir.index) 


#Transformação dos Dados

def transformar_para_inteiro (dataset_recebido2, coluna):
    dataframe=dataset_recebido2.copy()
    lista_valores_distintos = dataframe[coluna].unique().tolist()
    lista_valores_distintos=[vd for vd in lista_valores_distintos if pd.notna(vd)]
    eh_booleano = 'Yes' in lista_valores_distintos
    if eh_booleano:
        dataframe[coluna] = dataframe[coluna].replace({'No': 0, 'Yes': 1})
    else:
        if dataframe[coluna].isnull().any():
            dataframe[coluna]=dataframe[coluna].fillna(0)
        for i, valor in enumerate (lista_valores_distintos):
            dataframe[coluna] = dataframe[coluna].replace({valor: i+1})
    return dataframe

def para_inteiro_varias_colunas (dataset_recebido, lista_colunas):
    dataframe=dataset_recebido.copy()
    for i, valor in enumerate (lista_colunas):
        dataframe=transformar_para_inteiro(dataframe, valor)
    return dataframe


lista_nomes_colunas=['Offer', 'Internet Type', 'Contract', 'Payment Method', 'Gender',
                                                  'Married', 'Referred a Friend', 'Phone Service', 'Multiple Lines',
                                                    'Internet Service', 'Online Security', 'Online Backup',
                                                      'Device Protection Plan', 'Premium Tech Support', 'Streaming TV',
                                                        'Streaming Movies', 'Streaming Music', 'Unlimited Data', 
                                                        'Paperless Billing', 'Under 30', 'Senior Citizen', 'Married', 'Dependents',
                                                        'Churn Reason', 'Churn Category']
dsmutavel=para_inteiro_varias_colunas(dsmutavel, lista_nomes_colunas)




@st.cache_data
def ler_dataset():
    return dsmutavel

def filters_section():
    st.markdown("#### Filtros")
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