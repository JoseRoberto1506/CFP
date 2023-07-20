import streamlit as st
import plotly.express as px


def metodo_pagamento(df):
    fig = px.histogram(df, x='Payment Method')
    st.plotly_chart(fig)


def motivo_churn_x_satisfacao(df):
    fig = px.histogram(df, 
                       x='Customer Satisfaction', 
                       y='Churn Reason', 
                       color='Customer Satisfaction',)
    fig.update_layout(xaxis=dict(title='Count (sum)'))
    st.plotly_chart(fig)


def solicitacoes_serv_cliente(df):
    fig = px.scatter(df, x='Total Customer Svc Requests', y="Churn Value")
    st.plotly_chart(fig)


def tipo_contrato(df):
    fig = px.histogram(df, x='Contract')
    st.plotly_chart(fig)


def grafico_em_barra(df, opcao_de_grafico):
    df_grouped = df.groupby(f'{opcao_de_grafico}').size().reset_index(name='count')
    fig = px.bar(df_grouped, 
                 x=f'{opcao_de_grafico}', 
                 y='count',)
    st.plotly_chart(fig)
