import streamlit as st
import pandas as pd

st.set_page_config(
    page_title= "Home",
    page_icon= "🏠",
    layout= "wide",
    initial_sidebar_state= "auto",
    menu_items= {
        'About': "Desenvolvido por:\n- Dayvson Moura;\n- José Roberto;\n- Thales Mayrinck;\n- Vinícius Gustavo."
    }
)

@st.cache_data
def ler_dataset():
    return pd.read_csv("./data/Customer-Churn-Records.csv").head(100)


st.header("Olá, bem-vindo ao Customer Flight Predictor!")

# Informações sobre o CFP
st.markdown("""
            Com uma quantidade de bancos cada vez maior, as pessoas têm várias opções para decidirem onde colocar seu dinheiro, facilmente saindo de um banco e indo para outro que ofereça menores taxas e melhores serviços. Como a fuga de clientes impacta na receita e rentabilidade do banco, e é mais caro atrair novos clientes do que manter aqueles que já estão na base, é de grande interesse dos bancos identificar com antecedência quais clientes estão propensos a saírem, de forma que os gestores consigam elaborar estratégias para retê-los, a fim de garantir bons resultados e evitar a perca de <i>market share</i>.
            
            Diante dessa necessidade dos bancos, surgiu o Customer Flight Predictor. O CFP foi desenvolvido com o objetivo de prever se determinado cliente está propenso a encerrar sua conta no banco nos próximos meses, permitindo que a gestão do banco realize essa identificação prévia e evite o <i>churn</i> de clientes.
            """, 
            unsafe_allow_html= True)

# Mostrar o dataset utilizado
st.markdown("""
            ### Dataset
            Abaixo apresentamos o <i>dataset</i> utilizado no trabalho. Para obter mais informações sobre o que cada coluna representa e o seu tipo, verifique o [dicionário de dados](https://github.com/JoseRoberto1506/CFP/blob/main/data/Dicion%C3%A1rio%20de%20dados.pdf).
            """,
            unsafe_allow_html= True)
st.dataframe(ler_dataset())
