import pandas as pd
import streamlit as st
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import plotly.express as px


clustering_cols = ['Gender','Age','Married','Dependents']

st.set_page_config(
    page_title= "Agrupamento",
    page_icon= "🧠",
    layout= "wide",
    initial_sidebar_state= "collapsed",
    menu_items= {
        'About': "Desenvolvido por:\n- Dayvson Moura;\n- José Roberto;\n- Thales Mayrinck;\n- Vinícius Gustavo."
    }
)


def main():
    header()
    leitura = preprocessamento()
    clusterizado=clusterizacao(leitura,StandardScaler())
    graficos_iniciais_cluster(clusterizado)
    graficos_versus_servicos(clusterizado)



def header():
    st.header("Machine Learning")
    st.markdown("""
                Esta página apresenta os modelos de <i>machine learning</i> criados e os seus resultados.            
                """,
                unsafe_allow_html= True)


def preprocessamento():
    st.markdown("### Pré-processamento")
    st.markdown("""
                Nesta etapa, 
                """,
                unsafe_allow_html=True)
    df = ler_dataset()
    if st.checkbox("Mostrar dataset após o pré-processamento dos dados"):
        st.dataframe(df)
    st.divider()
    
    return df



@st.cache_data
def ler_dataset():
    df = pd.read_csv("./data/telco_churn_data.csv")
    return df.copy()


def transformacao(dfrecebido):
    df=dfrecebido.copy()
    st.markdown("### Transformação")
    st.markdown("""
                Nesta etapa, 
                """,
                unsafe_allow_html=True)
    
    colunas_categoricas_binarias = ['Married','Dependents','Under 30','Senior Citizen']
    df['Gender'].replace({'Male': 0, 'Female': 1}, inplace=True)
    for i, coluna in enumerate(colunas_categoricas_binarias):
        df[coluna].replace({'No': 0, 'Yes': 1}, inplace=True)

    
    
    if st.checkbox("Mostrar dataset após a transformação dos dados"):
        st.dataframe(df)
    st.divider()

    return df



def numero_clusters(df):
    st.markdown("### Número de clusters")
    with st.expander("Método do Cotovelo para a identificação da quantidade ótima de clusters"):
        cotovelo(df)
    with st.expander("Método da Silhueta para a identificação da quantidade ótima de clusters"):
        silhueta(df)


def clusterizacao(dfrecebido,scaler:TransformerMixin=None):
    st.markdown("### Clusterização")
    df_final_de_verdade=dfrecebido.copy()
    df_clusterizando_final=transformacao(df_final_de_verdade)
    df_clusterizando=df_clusterizando_final.copy()
    if scaler is not None:
        df_clusterizando = scale(df_clusterizando, scaler)
    modelo = KMeans(n_clusters=11, n_init=10, random_state=42)
    dfcluster=df_clusterizando[clustering_cols]
    numero_clusters(dfcluster) 
    dfcluster['Cluster']=modelo.fit_predict(dfcluster)
    df_final_de_verdade['Cluster']=dfcluster['Cluster']
    st.dataframe(df_final_de_verdade)

    return df_final_de_verdade



def scale(dfrecebido:pd.DataFrame, scaler:TransformerMixin):
    df=dfrecebido.copy()
    scaling_cols = [x for x in ['Age','Number of Dependents'] if x in clustering_cols]
    for c in scaling_cols:
        vals = df[[c]].values
        df[c] = scaler.fit_transform(vals)
    return df
    

def cotovelo(df):
    inertia_values = []
    k_values = range(2, 21)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, init='random', n_init=10, max_iter=300, random_state=42)
        kmeans.fit(df)
        inertia_values.append(kmeans.inertia_)

    plt.figure(figsize=(12, 10))
    plt.plot(k_values, inertia_values, marker='o')
    plt.xlabel('Número de clusters (K)')
    plt.ylabel('Inércia')
    plt.title('Método do Cotovelo para K-means')
    plt.xticks(k_values)
    st.pyplot(plt)

def silhueta(df):
    k_values = range(2, 21)
    selected_k = st.selectbox('Escolha o valor de K:', k_values)
    kmeans = KMeans(n_clusters=selected_k, init='random', n_init=10, max_iter=300, random_state=42)

    cluster_labels = kmeans.fit_predict(df)
    silhouette_avg = silhouette_score(df, cluster_labels)
    silhouette_values = silhouette_samples(df, cluster_labels)   

    plt.figure(figsize=(12, 10))    
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
    plt.ylabel('Clusters')
    plt.title(f'Análise das Silhuetas para K = {selected_k}')
    plt.yticks(range(selected_k), range(selected_k))
    plt.legend()
    st.pyplot(plt)

def graficos_iniciais_cluster(dfrecebido):
    df=dfrecebido.copy()
    fig = px.scatter (df, x='Cluster', y='Age')
    st.plotly_chart(fig)


    df_counts = df.groupby(['Cluster', 'Gender']).size().reset_index(name='Count')
    fig = px.bar(df_counts, x='Cluster', y='Count',color='Gender',
             title='Distribuição de Gênero por Cluster',
             labels={'Cluster': 'Cluster', 'Count': 'Quantidade de Clientes'},
             color_discrete_map={'Male': 'blue', 'Female': 'pink'},
             barmode='group')
    st.plotly_chart(fig)


    
    df_counts = df.groupby(['Cluster', 'Married']).size().reset_index(name='Count')
    fig = px.bar(df_counts, x='Cluster', y='Count',color='Married',
             title='Distribuição de Casados por Cluster',
             labels={'Cluster': 'Cluster', 'Count': 'Quantidade de Clientes'},
             color_discrete_map={'No': 'white', 'Yes': 'Blue'},
             barmode='group')
    st.plotly_chart(fig)


    
    fig = px.bar(df, x='Cluster', y='Dependents', )#color='Gender', hover_data=['Dependents', 'Married']
    df_counts = df.groupby(['Cluster', 'Dependents']).size().reset_index(name='Count')
    fig = px.bar(df_counts, x='Cluster', y='Count',color='Dependents',
             title='Distribuição de Clientes com Dependentes por Cluster',
             labels={'Cluster': 'Cluster', 'Count': 'Quantidade de Clientes'},
             color_discrete_map={'No': 'white', 'Yes': 'Blue'},
             barmode='group')
    st.plotly_chart(fig)


def graficos_versus_servicos(dfrecebido):
    servicos=['Phone Service','Multiple Lines','Internet Service','Online Security',
     'Online Backup','Device Protection Plan','Premium Tech Support','Streaming TV',
     'Streaming Movies','Streaming Music']
    df=dfrecebido.copy()
    for i in range(len(servicos)):
        fig = px.bar(df, x='Cluster', y=servicos[i], )#color='Gender', hover_data=['Dependents', 'Married']
        df_counts = df.groupby(['Cluster', servicos[i]]).size().reset_index(name='Count')
        fig = px.bar(df_counts, x='Cluster', y='Count',color=servicos[i],
                title=f'Cluster X {servicos[i]}',
                labels={'Cluster': 'Cluster', 'Count': 'Quantidade de Clientes'},
                color_discrete_map={'No': 'white', 'Yes': 'Blue'},
                barmode='group')
        st.plotly_chart(fig)

# def build_graphics(df):
#     st.header("Gráficos relativos ao churn")
#     mapa_de_opcoes = {'Telefone': 'Phone Service',
#                       'Múltiplas linhas': 'Multiple Lines',
#                       'Internet': 'Internet Service',
#                       'Segurança Online': 'Online Security',
#                       'Backup Online': 'Online Backup',
#                       'Seguro Celular': 'Device Protection Plan',
#                       'Suporte Premium': 'Premium Tech Support',
#                       'TV Fechada': 'Streaming TV',
#                       'Streaming de Filmes': 'Streaming Movies',
#                       'Streaming de Música': 'Streaming Music',
#                       }
#     grafico_selecionado = st.selectbox("Selecione o gráfico:",
#                                          ["Telefone",
#                                           "Múltiplas linhas",
#                                           "Internet",
#                                           "Segurança Online",
#                                           "Backup Online",
#                                           "Seguro Celular",
#                                           "Suporte Premium",
#                                           "TV Fechada",
#                                           "Streaming de Filmes",
#                                           "Streaming de Música"
#                                           ])


main()