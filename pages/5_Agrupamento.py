import pandas as pd
import streamlit as st
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler,LabelEncoder
import plotly.express as px
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots


clustering_cols = ['Age','Gender','Married','Dependents']

st.set_page_config(
    page_title= "Agrupamento",
    page_icon= "🗃",
    layout= "wide",
    initial_sidebar_state= "collapsed",
    menu_items= {
        'About': "Desenvolvido por:\n- Dayvson Moura;\n- José Roberto;\n- Thales Mayrinck;\n- Vinícius Gustavo."
    }
)


def main():
    header()
    df = ler_dataset()
    clusterizado = clusterizacao(df, StandardScaler())
    clusterizado_dois=clusterizacao_interativa()
    graficos_iniciais_cluster(clusterizado_dois)
    graficos_versus_servicos(clusterizado_dois)
    if st.checkbox("Mostrar dataset após clusterização interativa"):
        st.dataframe(clusterizado_dois)
    


def header():
    st.header("Clusterização")
    st.markdown("""
                Esta página apresenta a clusterização realizada e os gráficos contruídos para análise dos <i>clusters</i>.           
                """,
                unsafe_allow_html= True)


@st.cache_data
def ler_dataset():
    df = pd.read_csv("./data/telco_churn_data.csv")
    return df.copy()


def clusterizacao(dfrecebido, scaler:TransformerMixin=None):
    df_final_de_verdade = dfrecebido.copy()
    df_clusterizando_final = transformacao(df_final_de_verdade)
    df_clusterizando = df_clusterizando_final.copy()
    if scaler is not None:
        df_clusterizando = scale(df_clusterizando, scaler)
    dfcluster = df_clusterizando[clustering_cols]
    numero_clusters(dfcluster)
    modelo = KMeans(n_clusters=11, n_init=10, random_state=42)
    dfcluster['Cluster'] = modelo.fit_predict(dfcluster)
    df_final_de_verdade['Cluster'] = dfcluster['Cluster']

   

    return df_final_de_verdade


def transformacao(dfrecebido):
    df = dfrecebido.copy()
    st.markdown("### Transformação")
    st.markdown("""
                Nesta etapa, as colunas 'Married', 'Dependents', 'Under 30' e 'Senior Citizen' tiveram seus valores 'No' e 'Yes' convertidos para 0 e 1, respectivamente. Na coluna 'Gender', o valor 'Male' foi convetido para 0 e 'Female' para 1.
                """,
                unsafe_allow_html=True)
    
    colunas_categoricas_binarias = ['Married','Dependents','Under 30','Senior Citizen']
    df['Gender'].replace({'Male': 0, 'Female': 1}, inplace=True)
    for coluna in colunas_categoricas_binarias:
        df[coluna].replace({'No': 0, 'Yes': 1}, inplace=True)

    if st.checkbox("Mostrar dataset após a transformação dos dados"):
        st.dataframe(df)
    st.divider()

    return df


def scale(dfrecebido:pd.DataFrame, scaler:TransformerMixin):
    df=dfrecebido.copy()
    scaling_cols = [x for x in ['Age','Number of Dependents'] if x in clustering_cols]
    for c in scaling_cols:
        vals = df[[c]].values
        df[c] = scaler.fit_transform(vals)
    return df
    

def numero_clusters(df):
    st.markdown("### Número de clusters")
    st.markdown("""
                Nesta etapa foram utilizados o Método do Cotovelo e o Método da Silhueta para identificar a quantidade de <i>clusters</i> ideal para realizar o agrupamento.
                """,
                unsafe_allow_html=True)
    with st.expander("Método do Cotovelo para a identificação da quantidade ótima de clusters"):
        cotovelo(df)
    with st.expander("Método da Silhueta para a identificação da quantidade ótima de clusters"):
        silhueta(df)


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
    st.markdown("### Distribuição de caracterísicas dos clientes por Cluster")
    expander=st.expander("Visualizar")
    cols = expander.columns(1)
    df=dfrecebido.copy()
    fig = px.scatter (df, x='Cluster', y='Age')
    cols[0].plotly_chart(fig, use_container_width=False)
    df_counts = df.groupby(['Cluster', 'Gender']).size().reset_index(name='Count')
    fig = px.bar(df_counts, x='Cluster', y='Count',color='Gender',
             title='Distribuição de Gênero por Cluster',
             labels={'Cluster': 'Cluster', 'Count': 'Quantidade de Clientes'},
             color_discrete_map={'Male': 'blue', 'Female': 'pink'},
             barmode='group')
    cols[0].plotly_chart(fig, use_container_width=False)

    df_counts = df.groupby(['Cluster', 'Married']).size().reset_index(name='Count')
    fig = px.bar(df_counts, x='Cluster', y='Count',color='Married',
             title='Distribuição de Casados por Cluster',
             labels={'Cluster': 'Cluster', 'Count': 'Quantidade de Clientes'},
             color_discrete_map={'No': 'Red', 'Yes': 'Blue'},
             barmode='group')
    cols[0].plotly_chart(fig, use_container_width=False)

    fig = px.bar(df, x='Cluster', y='Dependents', )
    df_counts = df.groupby(['Cluster', 'Dependents']).size().reset_index(name='Count')
    fig = px.bar(df_counts, x='Cluster', y='Count',color='Dependents',
             title='Distribuição de Clientes com Dependentes por Cluster',
             labels={'Cluster': 'Cluster', 'Count': 'Quantidade de Clientes'},
             color_discrete_map={'No': 'Red', 'Yes': 'Blue'},
             barmode='group')
    cols[0].plotly_chart(fig, use_container_width=False)

    


def graficos_versus_servicos(dfrecebido):
    st.markdown("### Distribuição de serviços contratados pelos clientes por Cluster")
    expander=st.expander("Visualizar")
    cols = expander.columns(1)
    servicos=['Phone Service','Multiple Lines','Internet Service','Online Security', 'Online Backup','Device Protection Plan', 
              'Premium Tech Support','Streaming TV', 'Streaming Movies','Streaming Music']
    df=dfrecebido.copy()
    for servico in servicos:
        fig = px.bar(df, x='Cluster', y=servico, )
        df_counts = df.groupby(['Cluster', servico]).size().reset_index(name='Count')
        fig = px.bar(df_counts, x='Cluster', y='Count',color=servico,
                title=f'Cluster X {servico}',
                labels={'Cluster': 'Cluster', 'Count': 'Quantidade de Clientes'},
                barmode='group')
        
        cols[0].plotly_chart(fig, use_container_width=False)
    

def clusterizacao_interativa():
    data = pd.read_csv("./data/telco_churn_data.csv")
    clusterizando=data.copy()

    clustering_cols = ['Age', 'Gender', 'Married', 'Dependents']

    colunas_categoricas_binarias = ['Married','Dependents']
    clusterizando['Gender'].replace({'Male': 0, 'Female': 1}, inplace=True)
    for coluna in colunas_categoricas_binarias:
        clusterizando[coluna].replace({'No': 0, 'Yes': 1}, inplace=True)

    # Barra lateral para seleção de coluna e número de clusters
    st.header('Clusterização Interativa')
    
    coluna_cluster = st.multiselect('Clusterizar por:', clustering_cols, ['Age','Gender','Married','Dependents'])
    num_clusters = st.slider('Selecione o número de clusters:', 2, 15, 11)
    

    
    # Aplicar o algoritmo de agrupamento K-Means
    model = KMeans(n_clusters=num_clusters, random_state=0)
    clusterizando=scale(clusterizando,StandardScaler())
    clusterizando['Cluster'] = model.fit_predict(clusterizando[coluna_cluster])
    data['Cluster']=clusterizando['Cluster']

    
    
    return data

main()
