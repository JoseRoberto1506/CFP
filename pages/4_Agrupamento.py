import pandas as pd
import streamlit as st
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import plotly.express as px


st.set_page_config(
    page_title= "Agrupamento de Dados",
    page_icon= "",
    layout= "wide",
    initial_sidebar_state= "collapsed",
    menu_items= {
        'About': "Desenvolvido por:\n- Dayvson Moura;\n- José Roberto;\n- Thales Mayrinck;\n- Vinícius Gustavo."
    }
)


def main():
    header()
    df_processado = preprocessamento()
    features, rotulos = transformacao(df_processado)
    mineracao_de_dados(features, rotulos)


def header():
    st.header("Agrupamento de Dados")
    st.markdown("""
                Esta página apresenta algumas opções de agrupamento de dados a partir do <i>dataset</i>.
                """,
                unsafe_allow_html= True)


def preprocessamento():
    st.markdown("### Pré-processamento")
    df = ler_dataset()
    df = df.drop(['Customer ID', 'Churn Reason', 'Churn Category'], axis = 1)

    if st.checkbox("Mostrar dataset após o pré-processamento dos dados"):
        st.dataframe(df)
    st.divider()
    
    return df


@st.cache_data
def ler_dataset():
    df = pd.read_csv("./data/telco_churn_data.csv")
    return df.copy()


def transformacao(df):
    st.markdown("### Transformação")
    lista_nomes_colunas = ['Offer', 'Internet Type', 'Contract', 'Payment Method', 'Gender', 'Married', 
                            'Referred a Friend', 'Phone Service', 'Multiple Lines', 'Internet Service', 
                            'Online Security', 'Online Backup', 'Device Protection Plan', 'Premium Tech Support', 
                            'Streaming TV', 'Streaming Movies', 'Streaming Music', 'Unlimited Data', 'Paperless Billing', 
                            'Under 30', 'Senior Citizen', 'Married', 'Dependents', 'City', ]
    dataframe = para_inteiro_varias_colunas(df, lista_nomes_colunas)
    dataframe['Customer Satisfaction'].fillna(0, inplace=True)

    y = df['Churn Value'] # Rótulos    
    x = df.drop('Churn Value', axis = 1) # Features
    X_res, y_res = SMOTE().fit_resample(x, y)
    df_balanceado = pd.DataFrame(X_res, columns=x.columns)
    df_balanceado['Churn Value'] = y_res
    
    if st.checkbox("Mostrar dataset após a transformação dos dados"):
        st.dataframe(df_balanceado)
    st.divider()

    return x, y


def para_inteiro_varias_colunas (dataset_recebido, lista_colunas):
    dataframe = dataset_recebido
    for coluna in lista_colunas:
        dataframe = transformar_para_inteiro(dataframe, coluna)

    return dataframe


def transformar_para_inteiro (dataset_recebido, coluna):
    dataframe = dataset_recebido
    lista_valores_distintos = dataframe[coluna].unique().tolist()
    lista_valores_distintos = [vd for vd in lista_valores_distintos if pd.notna(vd)]
    if 'Yes' in lista_valores_distintos:
        dataframe[coluna] = dataframe[coluna].replace({'No': 0, 'Yes': 1})
    else:
        if dataframe[coluna].isnull().any():
            dataframe[coluna].fillna(0, inplace=True)
        for i, valor in enumerate(lista_valores_distintos):
            dataframe[coluna].replace({valor: i+1}, inplace=True)

    return dataframe


def mineracao_de_dados(x, y):
    st.markdown("### Mineração de dados")
    with st.expander("Random Forest"):
        random_forest(x, y)
        

def random_forest(x, y):
    st.markdown("""
                O Random Forest é um algoritmo de aprendizagem supervisionada que se baseia nas árvores de decisão, utilizado em problemas de classificação e regressão. De uma forma geral, ele seleciona aleatoriamente um subconjunto de características e combina as previsões individuais de várias árvores de decisão para obter uma previsão final mais precisa. Dentre as vantagens de utilizar o algoritmo de Random Forest estão a alta precisão, tolerância a dados ausentes e <i>outliers</i>, consegue lidar com grandes conjuntos de dados e tem um risco reduzido de <i>overfitting</i>. No entanto, é um algoritmo que requer mais recursos de armazenamento e tem um tempo de processamento maior.
                """, 
                unsafe_allow_html=True)
    st.markdown("#### Resultados")
    X_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) # Dividir o dataset para treino e teste
    classifier = RandomForestClassifier(n_estimators=100, random_state=42) # Criar o modelo de Random Forest
    classifier.fit(X_train, y_train) # Treinar o modelo
    y_pred = classifier.predict(x_test) # Fazer previsões com o modelo
    report = classification_report(y_test, y_pred, output_dict=True) # Relatório de classificação do Random Forest
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)
    feature_importance_rf(classifier, X_train)


def feature_importance_rf(classifier, x_train):
    st.markdown("#### Atributos mais importantes")
    importance = classifier.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': x_train.columns, 
                                          'Importance': importance}).sort_values(by='Importance', 
                                                                                 ascending=True).tail(5)
    fig = px.bar(feature_importance_df, 
                 y='Feature', 
                 x='Importance', 
                 labels={'Feature': 'Atributo', 'Importance': 'Importância'})
    st.plotly_chart(fig)


main()