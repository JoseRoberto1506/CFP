import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle


def main():
    df_processado = pre_processamento()
    features, rotulos = transformacao(df_processado)
    random_forest(features, rotulos)


def pre_processamento():
    df = ler_dataset()
    df = df[['Contract', 'Customer Satisfaction', 
            'Total Customer Svc Requests', 'Monthly Charge', 
            'Number of Referrals', 'Payment Method', 'Churn Value']]
    df['Customer Satisfaction'].fillna(df['Customer Satisfaction'].mode()[0], inplace=True)

    return df


def ler_dataset():
    df = pd.read_csv("./data/telco_churn_data.csv")
    return df.copy()


def transformacao(df):
    df = pd.get_dummies(df, columns=['Contract', 'Payment Method'])

    # Balanceamento dos dados
    y = df['Churn Value'] # Rótulos    
    x = df.drop('Churn Value', axis = 1) # Features
    X_res, y_res = SMOTE().fit_resample(x, y)
    df_balanceado = pd.DataFrame(X_res, columns=x.columns)
    df_balanceado['Churn Value'] = y_res
    
    return df_balanceado.drop('Churn Value', axis = 1), df_balanceado['Churn Value']


def random_forest(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)
    with open('model.pkl', 'wb') as file:
        pickle.dump(classifier, file)


main()