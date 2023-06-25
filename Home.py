import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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
    return pd.read_csv("./data/Customer-Churn-Records.csv")


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


## Gráfico 1: Torta 'Saiu x Não saiu':

# Falta configurar a exibição deles na página

#criando o dataframe principal a ser utilizado por literalmente TUDO
df = ler_dataset()

#contando os valores na coluna de 'Saiu'
contagem_torta_saiu = df['Exited'].value_counts()

fig, ax = plt.subplots(figsize=(8,6))
#Abaixo: uma lista de rótulos passada como argumento para o método 
rotulos_torta_saiu = ['Não saiu', 'Saiu'] 
#Abaixo: pandas+matplot #autopct mostra as porcentagens, labels nomeia os valores
contagem_torta_saiu.plot.pie(ax=ax, autopct='%1.1f%%', labels=rotulos_torta_saiu) 
#Abaixo: legenda superior
ax.legend(rotulos_torta_saiu) 
#Abaixo: título que fica do lado esquerdo
ax.set_ylabel('Status') 
fig.set_size_inches(8,6)

# mostra o gráfico no Streamlit
st.pyplot(fig) #streamlit que pega matplot




## Gráfico 3: Barras mostrando o percentual de cada score de satisfação:

#criando um dataframe com os valores de cada avaliação
contagem_barras_satisfacao = df['Satisfaction Score'].value_counts()
contagem_barras_satisfacao = contagem_barras_satisfacao.sort_index()

fig, ax = plt.subplots(figsize=(8,6))
contagem_barras_satisfacao.plot.barh()

ax.set_title ("Satisfação dos Clientes")
ax.set_xlim (0, 2500)

#adiciona os valores à direita das barras
for i, valor in enumerate (contagem_barras_satisfacao): 
    ax.annotate(str(valor), (valor, i), va='center') 

st.pyplot(fig)






## Gráfico 4: Grupos Reclamou x Saiu

#criando pequenos dataframes
reclamou_saiu = df[(df['Complain'] == 1) & (df['Exited'] == 1)].shape[0]
reclamou_nao_saiu = df[(df['Complain'] == 1) & (df['Exited'] == 0)].shape[0]
nao_reclamou_saiu = df[(df['Complain'] == 0) & (df['Exited'] == 1)].shape[0]
nao_reclamou_nao_saiu = df[(df['Complain'] == 0) & (df['Exited'] == 0)].shape[0]

#unindo eles num dicionário
contagem_barras_grupos = {'Status': ['Reclamou e saiu.', 'Reclamou e não saiu.', 'Não reclamou e saiu.', 'Não reclamou e não saiu.'],
         'Count': [reclamou_saiu, reclamou_nao_saiu, nao_reclamou_saiu, nao_reclamou_nao_saiu]}

#criando um dataframe do dicionário
df_grupos = pd.DataFrame(contagem_barras_grupos)
df_grupos = df_grupos.sort_values (by='Count')

fig, ax = plt.subplots(figsize=(8, 6))

#dizendo os nomes e os valores das colunas
ax.bar(x=df_grupos['Status'], height=df_grupos['Count'])
ax.set_title("Reclamou x Saiu")

#escrevendo os valores em cima das colunas
for i, v in enumerate(df_grupos['Count']):
    ax.text(i, v, str(v), ha='center', va='bottom')

st.pyplot(fig)


# Documentação
# df = DataFrame(pandas)
# contagem = Série (pandas)(não é objeto?)
# fig = Figura do gráfico (matplotlib)
# ax = Eixos do gráfico (matplotlib)
# 1-contagem dos valores com o pandas
# 2-cria o gráfico do matplot
# 3-indica com o pandas+matplot como é o gráfico
# 4-configura as coisas do gráfico com o matplot
# 5-mostra ele