import pandas as pd
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import numpy as np


st.set_page_config(
    page_title= "Cross Validation",
    page_icon= "⚔",
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
    resultados = cross_validation(features, rotulos)
    construir_dataframe_de_resultados(resultados)


def header():
    st.header("K-Fold Cross-Validation")
    st.markdown("""
                Esta página apresenta a média dos resultados das métricas de classificação dos modelos Random Forest, SVM, KNN, Naive Bayes e XGBoost após a execução do <i>5-fold cross-validation</i>.            
                """,
                unsafe_allow_html= True)


def preprocessamento():
    df = ler_dataset()
    df = df.drop(['Customer ID', 'City', 'Zip Code', 'Latitude', 'Longitude', 'Population', 'Churn Reason', 'Churn Category'], axis = 1)
    df['Customer Satisfaction'].fillna(df['Customer Satisfaction'].mode()[0], inplace=True)
    return df


@st.cache_data
def ler_dataset():
    df = pd.read_csv("./data/telco_churn_data.csv")
    return df.copy()


def transformacao(df):
    colunas_categoricas_binarias = ['Referred a Friend', 'Phone Service', 'Multiple Lines', 'Internet Service', 'Online Security', 
                                    'Online Backup', 'Device Protection Plan', 'Premium Tech Support', 'Streaming TV', 'Streaming Movies', 
                                    'Streaming Music', 'Unlimited Data', 'Paperless Billing', 'Under 30', 'Senior Citizen', 'Married', 
                                    'Dependents']
    df['Gender'].replace({'Male': 0, 'Female': 1}, inplace=True)
    for coluna in colunas_categoricas_binarias:
        df[coluna].replace({'No': 0, 'Yes': 1}, inplace=True)
    colunas_categoricas_multivalor = ['Offer', 'Internet Type', 'Contract', 'Payment Method']
    df = pd.get_dummies(df, columns=colunas_categoricas_multivalor)

    # Balanceamento dos dados
    y = df['Churn Value'] # Rótulos    
    x = df.drop('Churn Value', axis = 1) # Features
    X_res, y_res = SMOTE().fit_resample(x, y)
    df_balanceado = pd.DataFrame(X_res, columns=x.columns)
    df_balanceado['Churn Value'] = y_res
    
    return df_balanceado.drop('Churn Value', axis = 1), df_balanceado['Churn Value']


def cross_validation(x, y):
    st.markdown("### Resultados do 5-Fold Cross-Validation")
    modelos = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='linear'),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "Naive Bayes": GaussianNB(),
        "XGBoost": XGBClassifier(n_estimators=50, learning_rate=0.3, random_state=42)
    }
    metricas = ['accuracy', 'precision', 'recall', 'f1']
    resultados_modelos = []

    for nome_modelo, modelo in modelos.items():
        resultados_metricas = {'Modelo': nome_modelo}
        if nome_modelo in ["SVM", "KNN", "Naive Bayes"]:
            scaler = StandardScaler()
            x = scaler.fit_transform(x)
        cv_results = cross_validate(modelo, x, y, cv=5, scoring=metricas)
        for metrica, resultado in cv_results.items():
            resultados_metricas[metrica] = np.mean(resultado)
        resultados_modelos.append(resultados_metricas)

    return resultados_modelos

    
def construir_dataframe_de_resultados(resultados):
    metricas_df = pd.DataFrame(resultados)
    st.dataframe(metricas_df)


main()