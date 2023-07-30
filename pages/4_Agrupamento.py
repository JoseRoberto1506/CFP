import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler


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
    st.header("Agrupamento de Dados")
    st.markdown("""
                Esta página apresenta algumas opções de agrupamento de dados a partir do <i>dataset</i>.
                """,
                unsafe_allow_html= True)

ds = pd.read_csv("./data/telco_churn_data.csv")
dsmutavel = ds.copy()

#Limpeza dos Dados

dsmutavel = dsmutavel.drop('Customer ID', axis = 1)

dsmutavel = dsmutavel.drop('City', axis = 1)

dsmutavel = dsmutavel.drop('Latitude', axis = 1)

dsmutavel = dsmutavel.drop('Longitude', axis = 1)

dsmutavel = dsmutavel.drop('Zip Code', axis = 1)


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

dsmutavel['Customer Satisfaction']=dsmutavel['Customer Satisfaction'].fillna(0)



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

def header2():
    st.header("Método do Cotovelo")


def cotovelo():
    inertia_values = []

    k_values = range(1, 11)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(dsmutavel)
        inertia_values.append(kmeans.inertia_)

    plt.plot(k_values, inertia_values, marker='o')
    plt.xlabel('Número de clusters (K)')
    plt.ylabel('Inércia')
    plt.title('Método do Cotovelo para K-means')
    plt.xticks(k_values)
    plt.show()
    st.pyplot(plt)

def header3():
    st.header("Análise da Silhueta")


def silhueta():
    k_values = range(2, 11)

    plt.figure()
    plt.close()

    selected_k = st.selectbox('Escolha o valor de K:', k_values)

    
    kmeans = KMeans(n_clusters=selected_k, random_state=42)
    cluster_labels = kmeans.fit_predict(dsmutavel)
    
    silhouette_avg = silhouette_score(dsmutavel, cluster_labels)

    silhouette_values = silhouette_samples(dsmutavel, cluster_labels)
    
    plt.figure(figsize=(6, 4))
    
    y_lower = 10
    for i in range(selected_k):
        cluster_silhouette_values = silhouette_values[cluster_labels == i]
        cluster_silhouette_values.sort()
        
        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_values, alpha=0.7)
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            
        y_lower = y_upper + 10
    

    plt.axvline(x=silhouette_avg, color='red', linestyle='--', label='Média', linewidth=2)
    
    plt.xlabel('Valores do coeficiente de silhueta')
    plt.ylabel('Cluster')
    plt.title('Análise das Silhuetas para K = {}'.format(selected_k))
    plt.yticks([])
    plt.xticks(np.arange(-0.1, 1.1, 0.1))
    plt.xlim(-0.1, 1)
    plt.legend()
    st.pyplot(plt)

def header4():
    st.header("Clusterização")

def clusterizacao():
    colunas_numericas = dsmutavel.select_dtypes(include=[np.number]).columns.tolist()

# Caixas de seleção para escolher as duas colunas para clusterização
    coluna_x = st.selectbox('Selecione a primeira coluna:', colunas_numericas)
    coluna_y = st.selectbox('Selecione a segunda coluna:', colunas_numericas)

# Criar DataFrame apenas com as colunas selecionadas
    data_selected = dsmutavel[[coluna_x, coluna_y]]

# Aplicar o algoritmo KMeans aos dados
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(data_selected)

# Obtendo as etiquetas dos pontos (clusters)
    labels = kmeans.labels_

# Adicionando as etiquetas dos clusters ao DataFrame
    data_selected['Cluster'] = labels

# Plotando os dados coloridos de acordo com os clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(data_selected[coluna_x], data_selected[coluna_y], c=labels, cmap='viridis', alpha=0.6, s=100, label='Clusters')

# Plotando todos os dados do DataFrame em cinza para visualizar a dispersão completa
    plt.scatter(dsmutavel[coluna_x], dsmutavel[coluna_y], c='gray', alpha=0.4, s=20, label='Dados Originais')

    plt.xlabel(coluna_x)
    plt.ylabel(coluna_y)
    plt.title('Clusterização do DataFrame dsmutavel usando K-Means')
    plt.colorbar(label='Clusters')
    plt.legend()
    plt.show()

# Exibindo o plot no Streamlit
    st.pyplot()

def header4():
    st.header("Clusterização")

def clusterizacao():
    colunas_numericas = dsmutavel.select_dtypes(include=[np.number]).columns.tolist()

    coluna_x = st.selectbox('Selecione a primeira coluna:', colunas_numericas)
    coluna_y = st.selectbox('Selecione a segunda coluna:', colunas_numericas)

    data_selected = dsmutavel[[coluna_x, coluna_y]]

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(data_selected)

    labels = kmeans.labels

    data_selected['Cluster'] = labels

    plt.figure(figsize=(10, 6))
    plt.scatter(data_selected[coluna_x], data_selected[coluna_y], c=labels, cmap='viridis', alpha=0.6, s=100, label='Clusters')

    plt.scatter(dsmutavel[coluna_x], dsmutavel[coluna_y], c='gray', alpha=0.4, s=20, label='Dados Originais')

    plt.xlabel(coluna_x)
    plt.ylabel(coluna_y)
    plt.title('Clusterização do DataFrame dsmutavel usando K-Means')
    plt.colorbar(label='Clusters')
    plt.legend()
    plt.show()

    st.pyplot()


header()
filters_section()
header2()
cotovelo()
header3()
silhueta()
header4()
clusterizacao()


header4()
clusterizacao()