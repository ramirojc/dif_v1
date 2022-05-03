import streamlit as st
import pandas as pd
import numpy as np
import time

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import plotly.express as px
from streamlit_plotly_events import plotly_events
from PIL import Image

st.set_page_config(layout="wide")

ft_sel = ['BI_IVA', 'DJC_M_BASE_IMPONIBLE_CBA', 'DJC_M_IMPUESTO_DETERMINADO', 'DJC_M_PERCEPCIONES', 'DJC_M_RECAUDACIONES',
    'DJC_M_RETENCIONES', 'DJC_M_SALDO_A_PAGAR_DDJJ', 'DJC_M_SALDO_FAVOR_CONTRIBUYENTE', 'DJC_M_TOTAL_INGRESOS_NO_GRAV', 'Delta_BI']

drop_ft = ['DJC_ID_DIM_SUJETO', 'DJC_ID_DIM_PERIODO_CUOTA', 'DJC_ROL','5_Cl', '5_Cl_Dis', '10_Cl', '10_Cl_Dis', '20_Cl', '20_Cl_Dis']

@st.cache(allow_output_mutation=True)
def load_escenario_file(file):
    xls = pd.ExcelFile(file)
    raw_sample = pd.read_excel(xls, 'Muestra Dataset')
    pp_sample = pd.read_excel(xls, 'Muestra')
    escenario_data = pd.read_excel(xls, 'Diccionario',skiprows = 9, usecols='B:D')

    return raw_sample, pp_sample, escenario_data

# Setup SideBar
image = Image.open('calden-consultoria.png')
st.sidebar.image(image, caption='Calden Consultoria')

st.sidebar.title('Proyecto Cruce IIBB e IVA')

st.sidebar.title('Cargar datos de escenario')
uploaded_file = st.sidebar.file_uploader("Seleccione el escenario a cargar")

side_process = ''

if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    data_raw, data, escenario = load_escenario_file(uploaded_file)

    pp_sample_cols = data.columns.drop(drop_ft)

    data_num = data[pp_sample_cols]
    data_num = data_num.fillna(0)

    scaler = StandardScaler()
    data_sc = scaler.fit_transform(data_num)

    pca = PCA(n_components=3)
    data_pca = pd.DataFrame(pca.fit_transform(data_sc),columns=['x','y','z'])

    data_tsne = pd.DataFrame(
        TSNE(n_components=3, learning_rate='auto', init='random').fit_transform(data_sc),
        columns=['x','y','z'])

    side_process = st.sidebar.selectbox('Secciones',
        ('Escenario', 'Analisis Clusters', 'Analisis Grupo BI', 'Analisis Cluster Individuales', 'Analisis Contribuyentes'))


if side_process == 'Escenario':

    st.title('Escenario Seleccionado')
    st.subheader('Diccionario de variables utilizadas en escenario')

    st.dataframe(escenario)
    
elif side_process == 'Analisis Clusters':
    st.title('Analisis de clusters')
    
    # Figures
    st.subheader('Clustering PCA 3D Plot')

    fig_pca = px.scatter_3d(
        data_pca, x='x', y='y', z='z',
        color= data['5_Cl'].astype(str).values,
        hover_name= 'ID: ' + data.DJC_ID_DIM_SUJETO.astype(str),
        hover_data= {'BI CBA':data.BI_CBA, 'BI IVA': data.BI_IVA},
        template= 'seaborn')
    
    fig_pca.update_layout(
        autosize=False,
        showlegend=False,
        width=1200,
        height=600,
        margin=dict(
        l=0,
        r=0,
        b=0,
        t=0,
        pad=1
    ))

    
    selected_points = plotly_events(fig_pca, click_event=True, hover_event=False, select_event=False)
    st.write(selected_points)

    st.subheader('Clustering PCA-TSNE 3D Plot')
    
    fig_tsne = px.scatter_3d(
        data_tsne, x='x', y='y', z='z',
        hover_name= 'ID: ' + data.DJC_ID_DIM_SUJETO.astype(str),
        hover_data= {'BI CBA': data.BI_CBA, 'BI IVA': data.BI_IVA},
        color= data['5_Cl'].astype(str).values,
        template='seaborn')
    
    fig_tsne.update_coloraxes(showscale=False)
    fig_tsne.update_layout(
        autosize=False,
        showlegend=False,
        width=1200,
        height=600,
        margin=dict(
        l=0,
        r=0,
        b=0,
        t=0,
        pad=1
    ))

    st.plotly_chart(fig_tsne)

    # Boxplot

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Boxplot Base Imponible por Cluster')
        fig_box_BI = px.box(data, x="5_Cl", y="BI_CBA")
        st.plotly_chart(fig_box_BI)
    with col2:
        st.subheader('Boxplot Variacion de Base Imponible por Cluster')
        fig_box_dBI = px.box(data, x="5_Cl", y="Delta_BI")
        st.plotly_chart(fig_box_dBI)

    # Busqueda Anomalias

    st.subheader('Casos mas alejados de los centroides')

    for n in range(5):
        st.write(f'**Cluster {n}:**')
        st.dataframe(data[data['5_Cl']==n].sort_values('5_Cl_Dis', ascending=False).head(3))

elif side_process == 'Analisis Grupo BI':
    st.title('Analisis Grupo BI')

    max_bi = data.BI_CBA.max()
    st.subheader(f'Maxima BI en muestra: {max_bi}')

    n = st.number_input('Seleccione el numero de grupos', value=5)

    col_criteria = st.radio( "Seleccione la variable de clasificacion",
        ('BI IVA', 'BI CBA', 'Delta BI')) 

    if col_criteria == 'BI IVA':
        col = 'BI_IVA'
    elif col_criteria == 'BI CBA':
        col = 'BI_CBA'
    else:
        col = 'Delta_BI'

    bi_criteria = st.radio( "Criterio de agrupacion por BI",
        ('Monto Absoluto', 'Deciles')) 

    if bi_criteria == 'Deciles':
        data['qlabel'] = pd.qcut(data[col].values, n, labels=range(n))
    else:
        data['qlabel'] = pd.cut(data[col].values, n, labels=range(n))

    st.subheader('Grupos BI TSNE 3D Plot')
    
    fig_tsne = px.scatter_3d(
        data_tsne, x='x', y='y', z='z',
        hover_name= 'ID: ' + data.DJC_ID_DIM_SUJETO.astype(str),
        hover_data= {'BI CBA': data.BI_CBA, 'BI IVA': data.BI_IVA},
        color= data['qlabel'].astype(str).values,
        template='seaborn')
    
    fig_tsne.update_coloraxes(showscale=False)
    fig_tsne.update_layout(
        autosize=False,
        showlegend=False,
        width=1200,
        height=600,
        margin=dict(
        l=0,
        r=0,
        b=0,
        t=0,
        pad=1
    ))

    st.plotly_chart(fig_tsne)

    st.subheader('Grupos BI vs BI vs Delta BI')

    fig_cp = px.scatter(data, x="qlabel", y="BI_CBA", color="Delta_BI", hover_data=['DJC_ID_DIM_SUJETO'])
    fig_cp.update_layout(
        autosize=False,
        showlegend=False,
        width=1200,
        height=600,
        margin=dict(
        l=0,
        r=0,
        b=0,
        t=0,
        pad=1
    ))
    st.plotly_chart(fig_cp)

 

    
