import pandas as pd
import streamlit as st
import pickle


st.set_page_config(
    page_title= "Prever churn",
    page_icon= "🎯",
    layout= "wide",
    initial_sidebar_state= "collapsed",
    menu_items= {
        'About': "Desenvolvido por:\n- Dayvson Moura;\n- José Roberto;\n- Thales Mayrinck;\n- Vinícius Gustavo."
    }
)


st.header("Prever Churn de Cliente")

campo1, campo2, campo3 = st.columns(3)
with campo1:
    lista_contratos = ['Month-to-Month', 'One Year', 'Two Year']
    contrato_selecionado = st.selectbox('Tipo do contrato', lista_contratos)
with campo2:
    satisfacao = st.number_input('Satisfação do cliente', min_value=1, max_value=5)
with campo3:
    svc_requests = st.number_input('Solicitações ao serviço de atendimento ao cliente', value=0)
    
campo4, campo5, campo6 = st.columns(3)
with campo4:
    cobranca_mensal = st.number_input('Valor da conta mensal do cliente', min_value=0.00, value=0.00)
with campo5:
    indicacoes = st.number_input('Quantidade de indicações realizadas pelo cliente', min_value=0)
with campo6:
    metodos_pagamento = ['Bank Withdrawal', 'Credit Card', 'Mailed Check']
    metodo_pagamento = st.selectbox('Método de pagamento', metodos_pagamento)

user_input_df = pd.DataFrame({'Customer Satisfaction': [satisfacao], 
                              'Total Customer Svc Requests': [svc_requests], 
                              'Monthly Charge': [cobranca_mensal], 
                              'Number of Referrals': [indicacoes],})
for contrato in lista_contratos:
    if contrato == contrato_selecionado:
        user_input_df[f'Contract_{contrato}'] = 1
    else:
        user_input_df[f'Contract_{contrato}'] = 0
for met_pag in metodos_pagamento:
    if met_pag == metodo_pagamento:
        user_input_df[f'Payment Method_{met_pag}'] = 1
    else:
        user_input_df[f'Payment Method_{met_pag}'] = 0

with open('model.pkl', 'rb') as file:
    clf = pickle.load(file)

if st.button('Realizar Previsão'):
    previsao = clf.predict(user_input_df)
    if previsao == 1:
        st.error('Grande chance do cliente deixar a empresa nos próximos meses')
    else:
        st.success('Pouca chance do cliente deixar a empresa nos próximos meses')
