import pandas as pd
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


st.set_page_config(
    page_title= "Machine Learning",
    page_icon= "🧠",
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
    st.header("Machine Learning")
    st.markdown("""
                Esta página apresenta os modelos de <i>machine learning</i> criados e os seus resultados.            
                """,
                unsafe_allow_html= True)


def preprocessamento():
    st.markdown("### Pré-processamento")
    st.markdown("""
                Nesta etapa, a coluna 'Customer ID' foir removida, pois ela é apenas um identificador único para cada cliente e não tem impacto nos modelos de <i>machine learning</i> que serão utilizados. Também foram removidas as colunas 'Churn Reason' e 'Churn Category', visto que que elas podem impactar negativamente os resultados dos algoritmos pois indicam o motivo que levou determinados clientes a deixarem a empresa, fazendo com que os modelos usados decorem os clientes que sairão. Na coluna 'Customer Satisfaction', foi utilizada a moda para preencher os valores faltantes, identificando qual o valor que mais se repete e inserindo-o nas linhas que possuem valor nulo.
                """,
                unsafe_allow_html=True)
    df = ler_dataset()
    df = df.drop(['Customer ID', 'Churn Reason', 'Churn Category'], axis = 1)
    df['Customer Satisfaction'].fillna(df['Customer Satisfaction'].mode()[0], inplace=True)

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
    st.markdown("""
                Nesta etapa, foi utlizada a técnica <i>Label Encoding</i> para converter os dados das variáveis categóricas em valores numéricos. Em seguida, foi realizado o balanceamento dos dados utilizando a técnica <i>SMOTE</i>.
                """,
                unsafe_allow_html=True)
    lista_nomes_colunas = ['Offer', 'Internet Type', 'Contract', 'Payment Method', 'Gender', 'Married', 
                           'Referred a Friend', 'Phone Service', 'Multiple Lines', 'Internet Service', 
                           'Online Security', 'Online Backup', 'Device Protection Plan', 'Premium Tech Support', 
                           'Streaming TV', 'Streaming Movies', 'Streaming Music', 'Unlimited Data', 'Paperless Billing', 
                           'Under 30', 'Senior Citizen', 'Married', 'Dependents', 'City', ]
    for coluna in lista_nomes_colunas:
        df[coluna] = LabelEncoder().fit_transform(df[coluna])

    # Balanceamento dos dados
    y = df['Churn Value'] # Rótulos    
    x = df.drop('Churn Value', axis = 1) # Features
    X_res, y_res = SMOTE().fit_resample(x, y)
    df_balanceado = pd.DataFrame(X_res, columns=x.columns)
    df_balanceado['Churn Value'] = y_res
    
    if st.checkbox("Mostrar dataset após a transformação dos dados"):
        st.dataframe(df_balanceado)
    st.divider()

    return df_balanceado.drop('Churn Value', axis = 1), df_balanceado['Churn Value']


def mineracao_de_dados(x, y):
    st.markdown("### Mineração de dados")
    st.markdown("""
                Nesta etapa de mineração de dados, selecionamos os modelos Random Forest, SVM, KNN e Naive Bayes. Para todos os modelos, o <i>dataset</i> foi particionado em 80% para o conjunto de treinamento e 20% para o conjunto de teste, utilizando a semente (<i>seed</i>) 42.<br>
                Para a visualização dos resultados dos modelos de <i>machine learning</i> utilizados, basta selecionar abaixo o modelo desejado.
                """,
                unsafe_allow_html=True)
    with st.expander("Random Forest"):
        random_forest(x, y)
    with st.expander("SVM"):
        svm(x,y)
    with st.expander("KNN"):
        knn(x, y)
    with st.expander("Naive Bayes"):
        naive_bayes(x, y)
    

def random_forest(x, y):
    st.markdown("""
                O Random Forest é um algoritmo de aprendizagem supervisionada que se baseia nas árvores de decisão, utilizado em problemas de classificação e regressão. De uma forma geral, ele seleciona aleatoriamente um subconjunto de características e combina as previsões individuais de várias árvores de decisão para obter uma previsão final mais precisa. Dentre as vantagens de utilizar o algoritmo de Random Forest estão a alta precisão, tolerância a dados ausentes e <i>outliers</i>, consegue lidar com grandes conjuntos de dados e tem um risco reduzido de <i>overfitting</i>. No entanto, é um algoritmo que requer mais recursos de armazenamento e tem um tempo de processamento maior.
                """, 
                unsafe_allow_html=True)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    metricas_de_classificacao(y_test, y_pred, "Random Forest")
    feature_importance(X_train.columns, classifier.feature_importances_)
    matriz_de_confusao(confusion_matrix(y_test, y_pred))


def svm(x, y):
    st.markdown("""
                Support Vector Machines (SVM) são um conjunto de métodos de aprendizado de máquina supervisionado utilizados para classificação, regressão e detecção de outliers. De uma forma geral, ele busca encontrar um hiperplano que separe as amostras de diferentes classes no espaço vetorial, maximizando a margem entre as amostras mais próximas de cada classe e usando os vetores de suporte para determinar sua posição e orientação. As vantagens do SVM incluem sua efetividade em espaços de alta dimensionalidade, sua capacidade de lidar com mais dimensões do que amostras e o uso eficiente de um subconjunto de pontos de treinamento na função de decisão (chamados de vetores de suporte). Ele também é versátil, permitindo o uso de diferentes funções de <i>kernel</i> para a função de decisão.
                """, 
                unsafe_allow_html=True)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train_scaled, y_train)
    y_pred = svm_classifier.predict(X_test_scaled)
    metricas_de_classificacao(y_test, y_pred, "SVM")
    feature_importance(X_train.columns, svm_classifier.coef_[0])
    matriz_de_confusao(confusion_matrix(y_test, y_pred))


def metricas_de_classificacao(y_true, y_pred, modelo):
    st.markdown(f"<h6>Métricas de classificação do {modelo}</h6>", unsafe_allow_html=True)
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.style.format({'precision': '{:.2%}', 'recall': '{:.2%}', 'f1-score': '{:.2%}', 'support': '{:.0f}'})
    st.dataframe(report_df)


def feature_importance(features, importances):
    feature_importance_df = pd.DataFrame({'Feature': features, 
                                          'Importance': abs(importances)}).sort_values(by='Importance', 
                                                                                       ascending=True).tail(5)
    fig = px.bar(feature_importance_df, 
                 y='Feature', 
                 x='Importance', 
                 labels={'Feature': 'Atributo', 'Importance': 'Importância'},
                 title='Feature Importance')
    st.plotly_chart(fig)


def matriz_de_confusao(cm):
    rotulos_classes = ["Não Saiu", "Saiu"]
    df_cm = pd.DataFrame(cm, 
                         index=rotulos_classes, 
                         columns=rotulos_classes)
    fig = ff.create_annotated_heatmap(z=df_cm.values, 
                                      x=list(df_cm.columns), 
                                      y=list(df_cm.index), 
                                      colorscale='Blues')
    fig.update_layout(xaxis_title='Valores Previstos', 
                      yaxis_title='Valores Reais', 
                      title='Matriz de Confusão')
    st.plotly_chart(fig)


def knn(x, y):
    st.markdown("""
                O K-Nearest Neighbors (KNN) é um algoritmo de aprendizado de máquina supervisionado usado principalmente para tarefas de classificação e regressão. Ele se baseia no princípio de que exemplos semelhantes estão próximos uns dos outros no espaço de características. O KNN classifica um novo exemplo com base na maioria das classes dos seus vizinhos mais próximos.
                """, 
                unsafe_allow_html=True)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    metricas_de_classificacao(y_test, y_pred, "KNN")
    matriz_de_confusao(confusion_matrix(y_test, y_pred))


def naive_bayes(x, y):
    st.markdown("""
                O Naive Bayes é um algoritmo de aprendizado de máquina que usa o Teorema de Bayes para calcular a probabilidade de uma instância pertencer a uma classe específica com base nas probabilidades das características condicionadas à classe. Embora faça a suposição simplificada de independência entre as características, o Naive Bayes é eficaz em tarefas de classificação, como filtragem de spam e categorização de texto, sendo particularmente útil em cenários com grandes conjuntos de dados e características categóricas, apesar de suas limitações.
                """, 
                unsafe_allow_html=True)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    naive_bayes = GaussianNB()
    naive_bayes.fit(X_train, y_train)
    y_pred = naive_bayes.predict(X_test)
    metricas_de_classificacao(y_test, y_pred, "Naive Bayes")
    matriz_de_confusao(confusion_matrix(y_test, y_pred))


main()