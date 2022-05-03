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

st.set_page_config(
    page_title="Analisis de datos - DIF"
    layout="wide")

ft_sel = ['BI_IVA', 'DJC_M_BASE_IMPONIBLE_CBA', 'DJC_M_IMPUESTO_DETERMINADO', 'DJC_M_PERCEPCIONES', 'DJC_M_RECAUDACIONES',
    'DJC_M_RETENCIONES', 'DJC_M_SALDO_A_PAGAR_DDJJ', 'DJC_M_SALDO_FAVOR_CONTRIBUYENTE', 'DJC_M_TOTAL_INGRESOS_NO_GRAV', 'Delta_BI']

ft_num_sel = ['DJC_ID_DIM_PERIODO_CUOTA','BI_CBA','BI_IVA','Bancarizacion','Delta_BI','Delta_BI%','Delta_BI_anio','FormalidadCmpras']

drop_ft = ['DJC_ID_DIM_SUJETO', 'DJC_ID_DIM_PERIODO_CUOTA', 'DJC_ROL','5_Cl', '5_Cl_Dis', '10_Cl', '10_Cl_Dis', '20_Cl', '20_Cl_Dis']

@st.cache(allow_output_mutation=True)
def load_escenario_file(file):
    xls = pd.ExcelFile(file)
    raw_sample = pd.read_excel(xls, 'Muestra Dataset')
    pp_sample = pd.read_excel(xls, 'Muestra')
    escenario_data = pd.read_excel(xls, 'Diccionario',skiprows = 9, usecols='A:D')

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

    nclus_map = {
        '5 Clusters':'5_Cl',
        '10 Clusters':'10_Cl',
        '20 Clusters':'20_Cl',
    }

    # Seleccion clusterizacion
    nclus_dp = st.sidebar.selectbox('Numero de clusters',
        ('5 Clusters', '10 Clusters', '20 Clusters'))
    
    nclus = nclus_map[nclus_dp]

    side_process = st.sidebar.selectbox('Secciones',
        ('Escenario', 'Analisis Clusters', 'Analisis Grupo BI', 'Analisis Cluster Individuales', 'Analisis Contribuyentes'))


#-----------------------
# Main Column
#-----------------------

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
        color= data[nclus].astype(str).values,
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

    st.plotly_chart(fig_pca)
    #selected_points = plotly_events(fig_pca, click_event=True, hover_event=False, select_event=False)
    #st.write(selected_points)

    st.subheader('Clustering PCA-TSNE 3D Plot')
    
    fig_tsne = px.scatter_3d(
        data_tsne, x='x', y='y', z='z',
        hover_name= 'ID: ' + data.DJC_ID_DIM_SUJETO.astype(str),
        hover_data= {'BI CBA': data.BI_CBA, 'BI IVA': data.BI_IVA},
        color= data[nclus].astype(str).values,
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
        fig_box_BI = px.box(data, x=nclus, y="BI_CBA")
        st.plotly_chart(fig_box_BI)
    with col2:
        st.subheader('Boxplot Variacion de Base Imponible por Cluster')
        fig_box_dBI = px.box(data, x=nclus, y="Delta_BI")
        st.plotly_chart(fig_box_dBI)

    # Busqueda Anomalias

    st.subheader('Casos mas alejados de los centroides')

    for n in range(data[nclus].nunique()):
        st.write(f'**Cluster {n}:**')
        st.dataframe(data[data[nclus]==n].sort_values(f'{nclus}_Dis', ascending=False).head(3))

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

    for n in range(5):
        st.write(f'**Cluster {n}:**')
        st.dataframe(data[data['qlabel']==n].sort_values('Delta_BI', ascending=False).head(3))

elif side_process == 'Analisis Cluster Individuales':

    st.title('Analisis Individual de Cluster')
    st.subheader('Seleccione el cluster que desea analizar')

    #data = pd.read_csv('D:\DIF_CALDEN\Resultados\Esc_015_raw_sample.csv')

    cluster_sel = st.selectbox('cluster', data[nclus].sort_values().unique())

    #Metricas Generales
    
    data_sel = data[ft_num_sel]
    data_agg_median = data_sel.median().reset_index()

    data_sel_cl = data_sel[data[nclus] == cluster_sel]
    data_agg_median_cl = data_sel_cl.median().reset_index()

    st.subheader('Mediana y Variacion de mediana con respecto al valor global')

    st_columns = st.columns(4)

    for r in data_agg_median_cl.iterrows():
        cl_val = r[1][0]
        cl_delta = np.around((cl_val - data_agg_median.loc[r[0],0]) * 100 / cl_val, 2)

        st_columns[r[0]%4].metric(r[1]['index'], "%.3f"%cl_val, cl_delta)

    st.subheader('')

    col1, col2 = st.columns(2)
    
    col1.subheader('Histograma base imponible IVA')
    fig_cl_iva = px.histogram(data, x=["BI_IVA","BI_CBA"])
    col1.plotly_chart(fig_cl_iva)

    col2.subheader('Base IIBB vs Base IVA')
    fig_cl_dbi = px.scatter(data, x="BI_CBA", y='BI_IVA')
    col2.plotly_chart(fig_cl_dbi)

    col1.subheader('Histograma de distancias al centro del cluster')
    fig_cl_dis = px.histogram(data, x=f'{nclus}_Dis')
    col1.plotly_chart(fig_cl_dis)

    col2.subheader('Histograma Delta BI')
    fig_cl_act3d = px.histogram(data, x='Delta_BI')
    col2.plotly_chart(fig_cl_act3d)

elif side_process == 'Analisis Contribuyentes':

    st.title('Analisis Contribuyentes')
    st.header('Ingrese identificador de usuario')

    ft_con=[
        'DJC_ID_DIM_SUJETO', 'DJC_ID_DIM_PERIODO_CUOTA', 'DJA_COD_ACTIVIDAD',
        'BI_CBA', 'BI_IVA', 'Delta_BI',
        'DJC_M_BASE_IMPONIBLE_CBA', 'DJC_M_IMPUESTO_DETERMINADO', 'DJC_M_PERCEPCIONES','DJC_M_PERCEPCIONES_ADUANERAS',
        'DJC_M_RECAUDACIONES','DJC_M_RETENCIONES','DJC_M_SALDO_A_PAGAR_DDJJ','DJC_M_SALDO_FAVOR_CONTRIBUYENTE','DJC_M_TOTAL_INGRESOS_NO_GRAV',
        '5_Cl','10_Cl','20_Cl']

    data_sel = data_raw[ft_con]
    data_sel['PERIODO_date'] = data_sel.DJC_ID_DIM_PERIODO_CUOTA.astype(str)

    selected_user = st.selectbox('contribuyentes', tuple(data_sel.DJC_ID_DIM_SUJETO.unique()))

    filtered_data = data_sel[data_sel.DJC_ID_DIM_SUJETO == selected_user].sort_values('DJC_ID_DIM_PERIODO_CUOTA')
    filtered_data['source'] = f'Contribuyente: {selected_user}'

    con_cluster = filtered_data[nclus].mode()[0]
    con_activity = filtered_data.DJA_COD_ACTIVIDAD.mode()[0]
    con_bi = filtered_data.BI_CBA.median()

    # Same Cluster
    con_cluster_data = data_sel[(data_sel[nclus]==con_cluster) & (data_sel.DJC_ID_DIM_SUJETO!=selected_user)]
    con_cluster_data_ts = con_cluster_data.groupby('PERIODO_date').median().reset_index()
    con_cluster_data_ts['source'] = f'Cluster: {con_cluster} en {nclus}'

    cluster_comp_data = pd.concat([filtered_data, con_cluster_data_ts]).sort_values('PERIODO_date')

    # Same Activity
    con_act_data = data_sel[(data_sel.DJA_COD_ACTIVIDAD==con_activity) & (data_sel.DJC_ID_DIM_SUJETO!=selected_user)]
    con_act_data_ts = con_act_data.groupby('PERIODO_date').median().reset_index()
    con_act_data_ts['source'] = f'Actividad: {con_activity}'

    act_comp_data = pd.concat([filtered_data, con_act_data_ts]).sort_values('PERIODO_date')

    # Same Size
    con_bi_data = data_sel.groupby('DJC_ID_DIM_SUJETO').median().reset_index()
    bi_min = con_bi * 0.9
    bi_max = con_bi * 1.1

    con_bi_data_same = con_bi_data[(con_bi_data.BI_CBA > bi_min) & (con_bi_data.BI_CBA < bi_max)]
    same_bi_cons = con_bi_data_same.DJC_ID_DIM_SUJETO.to_list()

    con_bi_data = data_sel[data_sel.DJC_ID_DIM_SUJETO.isin(same_bi_cons)]

    con_bi_data_ts = con_bi_data.groupby('PERIODO_date').median().reset_index()
    con_bi_data_ts['source'] = f'BI entre {bi_min:.0f} y {bi_max:.0f}'

    bi_comp_data = pd.concat([filtered_data, con_bi_data_ts]).sort_values('PERIODO_date')
    

    # Show Data
    st.dataframe(filtered_data)

    st.subheader('Graficas Comparativas')

    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(filtered_data, x="PERIODO_date", y="BI_CBA", title='Base Imponible Vs Periodo')
        st.plotly_chart(fig)

        fig = px.line(cluster_comp_data, x="PERIODO_date", y="BI_CBA", color='source', title='Comparativo con mismo cluster')
        st.plotly_chart(fig)

        fig = px.line(act_comp_data, x="PERIODO_date", y="BI_CBA", color='source', title='Comparativo con misma actividad')
        st.plotly_chart(fig)

        fig = px.line(bi_comp_data, x="PERIODO_date", y="BI_CBA", color='source', title='Comparativo BI ±10%')
        st.plotly_chart(fig)

    with col2:
        fig = px.line(filtered_data, x="PERIODO_date", y="Delta_BI", title='Discrepancia Base Imponible Vs Periodo')
        st.plotly_chart(fig)
 
        fig = px.line(cluster_comp_data, x="PERIODO_date", y="Delta_BI", color='source', title='Comparativo con mismo cluster')
        st.plotly_chart(fig)

        fig = px.line(act_comp_data, x="PERIODO_date", y="Delta_BI", color='source', title='Comparativo con misma actividad')
        st.plotly_chart(fig)

        fig = px.line(bi_comp_data, x="PERIODO_date", y="Delta_BI", color='source', title='Comparativo BI ±10%')
        st.plotly_chart(fig)
