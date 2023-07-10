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
    return pd.read_csv("./data/telco_churn_data.csv").head(100)


def header():
    st.header("Olá, bem-vindo ao Customer Flight Predictor!")
    st.markdown("""
                Com o crescimento do mercado de telecomunicações, principalmente após achegada da tecnologia 5G, as empresas de telecomunicação têm investido cada vez mais em infraestrutura, visando melhorar seus serviços. Essa crescente competitividade permite que as pessoas saiam facilmente de uma empresa e vá para outra que tenha ofertas mais atrativas. 
                
                Como a migração de clientes impacta a receita e a rentabilidade das empresas, e atrair novos clientes é mais caro do que manter aqueles que já estão na base, é de grande interesse dessas empresas identificar com antecedência quais clientes estão propensos a cancelar seus serviços, de forma que os gestores consigam elaborar estratégias para retê-los, a fim de garantir bons resultados e evitar a perca de <i>market share</i>.

                Diante dessa necessidade das empresas de telecomunicação, surge o Customer Flight Predictor (CFP). O CFP foi desenvolvido com o objetivo de prever se um determinado cliente está propenso a cancelar seus serviços na empresa nos próximos meses, permitindo que a gestão da empresa realize essa identificação prévia e evite o <i>churn</i> de clientes.
                """, 
                unsafe_allow_html= True)


def about_dataset():
    st.markdown("""
                ### Dataset
                Abaixo apresentamos brevemente, mostrando apenas 100 linhas, o <i>dataset</i> utilizado no trabalho. Para obter mais informações sobre o que cada coluna representa e o seu tipo, verifique o [dicionário de dados](https://github.com/JoseRoberto1506/CFP/blob/main/data/Dicion%C3%A1rio%20de%20dados.pdf).
                """,
                unsafe_allow_html= True)
    st.dataframe(ler_dataset())


header()
about_dataset()
