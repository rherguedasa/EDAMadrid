#---------------------------------------------------------------------------LIBRERIAS----------------------------------------------------------------------------------------------------------------------------------

# librerias básicas
import numpy as np
import pandas as pd
import wget
import matplotlib.pyplot as plt
import requests
import os
import gzip
import json
import urllib.request  
import urllib
import warnings
import base64
warnings.simplefilter(action='ignore')
import random

# librerias machine learning para regresión lineal
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error
from sklearn import preprocessing

# mapas interactivos
import folium
from folium.plugins import FastMarkerCluster
import geopandas as gpd
from branca.colormap import LinearColormap
import pydeck as pdk

# plotear gráficos y visualización
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.io as pio
import folium
import plotly.express as px
from plotly.offline import plot
import chart_studio.plotly as py
from plotly.offline import iplot, init_notebook_mode
import seaborn as sns
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)
sns.set()
import streamlit as st
pio.templates.default = "plotly_dark"

# streamlit
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components
import google 
import base64
from PIL import Image

from pyngrok import ngrok
from IPython.display import display
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
import urllib.request
import unicodedata
from unicodedata import name
from streamlit.elements.utils import (check_callback_rules, check_session_state_rules, get_label_visibility_proto_value)

# editor de texto
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from wordcloud import WordCloud
from wordcloud import ImageColorGenerator
from PIL import Image
from wordcloud import STOPWORDS
nltk.download('stopwords')

#--------------------------------------------------------------------------CÓDIGO USADO EN EL PREPROCESAMIENTO-------------------------------------------------------------------------------------------------------

# cargamos el df procedente de un csv
df = pd.read_csv('airbnb_anuncios.csv')

# lista de columnas a eliminar
columns_to_drop = ["name","id", "host_name","last_review"]
# eliminamos las columnas anteriormente seleccionadas
df.drop(columns_to_drop, axis=1, inplace=True)

# ponemos a 0 los valores nulos de la columan 'revireviews_per_month'
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)


#----------------------------------------------------------------------------------COMENZAMOS LA APP-----------------------------------------------------------------------------------------------------------------
# establecemos la configuración de la página
st.set_page_config(page_title='EDA MADRID' , layout='centered', page_icon="")
image = Image.open('C:\\Users\\rober\\Documents\\Rober\\Bootcamp\\Modulo 2\\20-Trabajo Módulo 2\\datos\\alcalawide.png')
st.image(image, width=800)
st.write('Puerta de Alcalá: Imagen creada con DALL-E-2')
st.title('EDA AIRBNB: MADRID')

# creamos las pestañas que van a dividir nuestra app
tabs = st.tabs(['MADRID','EDA', 'CONCLUSIONES'])


# establecemos la imagen de fondo de la app
# además añadimos codigo CSS para eliminar el fondo
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
     <style>
        .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local("background.png")

# creamos una side bar y añadimos una barra con una pagina que nos muestra el tiempo en Madrid
st.sidebar.title("El tiempo en Madrid")
st.sidebar.write(f'<iframe src="https://www.accuweather.com/es/es/madrid/308526/hourly-weather-forecast/308526" width="" height="500" style="overflow:auto"></iframe>', unsafe_allow_html=True)
url = "https://www.accuweather.com/es/es/madrid/308526/hourly-weather-forecast/308526"
st.sidebar.markdown("[Accuweather](%s)" % url)

# creamos una side bar y añadimos una barra con una pagina que nos muestra las fiestas de los barrios en Madrid 
# añadimos código css para eliminar el fondo y que quede transparente
st.sidebar.title("Calendario de Fiestas")
st.sidebar.write(f'<iframe src="https://www.epe.es/es/madrid/20220803/fiestas-barrios-madrid-ps-13994794" width="" height="500" style="overflow:auto"></iframe>', unsafe_allow_html=True)
url = "https://www.epe.es/es/madrid/20220803/fiestas-barrios-madrid-ps-13994794"
st.sidebar.markdown("[Epe](%s)" % url)
st.markdown(
    f"""
    <style>
    [data-testid="stHeader"] {{
    background-color: rgba(0, 0, 0, 0);
    }}


    [data-testid="stSidebar"]{{                 
    background-color: rgba(0, 0, 0, 0);
    border: 0.5px solid #ff4b4b;
        }}

        
    [data-baseweb="tab"] {{
    background-color: rgba(0, 0, 0, 0);
        }}
    </style>
    """
, unsafe_allow_html=True)


#--------------------------------------------------------------------------------------------MADRID INTRO---------------------------------------------------------------------------------------------------------------

# primera pestaña
tab_plots = tabs[0]

# añadimos una descripción sobre Madrid y la situamos en el mapa
with tab_plots:
    st.header('MADRID')
    st.subheader('Madrid: Capital de España')
    with st.expander('**Información general sobre Madrid**'):
        st.write("""
                Madrid es la capital y ciudad más grande de España. Sus coordenadas son 40.4168° N, 3.7038° O. A 2021, 
                la población estimada de Madrid es de alrededor de 6.7 millones de personas. La área metropolitana de Madrid, que incluye la ciudad y los municipios circundantes, 
                tiene una población de alrededor de 6.5 millones de personas. Madrid es la tercera ciudad más poblada de la Unión Europea, después de Londres y Berlín. 
                La ciudad está ubicada en el centro de la Península Ibérica y es el centro político, económico y cultural de España.
                """)
    html = open("map.html", "r", encoding='utf-8').read()
    st.components.v1.html(html, width=800, height=600)

# adjuntamos una página web que nos situa los mejores sitios para ver de Madrid
with tab_plots:
    st.subheader('¿Que ver en Madrid?')
    with st.expander('**Madrid ciudad con historia**'):
        st.write("""
                Madrid es conocida por su rica historia, su cultura vibrante y su arquitectura impresionante. 
                La ciudad es el hogar de algunos de los museos más famosos del mundo, como el Museo del Prado, el Museo Reina Sofía y el Museo Thyssen-Bornemisza. 
                Además, Madrid es famosa por su vida nocturna y su gastronomía deliciosa. 
                En resumen Madrid es una ciudad llena de historia, cultura, arte y diversión, vale la pena visitar
                """)
    st.write(f'<iframe src="https://www.viajandoporelmundomundial.com/que-ver-en-madrid/" width="800" height="600" style="overflow:auto"></iframe>', unsafe_allow_html=True)
    url = "https://www.viajandoporelmundomundial.com/que-ver-en-madrid/"
    st.markdown("[Viajandoporelmundomundial](%s)" % url)


#----------------------------------------------------------------------------------EDA---------------------------------------------------------------------------------------------------------------

# segunda pestaña
tab_plots = tabs[1]

# creamos una matriz de coorrelación entre varias variables
with tab_plots:
    corr = df[['host_id', 'neighbourhood_group', 'neighbourhood', 'latitude',
            'longitude', 'room_type', 'price', 'minimum_nights',
            'number_of_reviews', 'reviews_per_month',
            'calculated_host_listings_count', 'availability_365']].corr()

# creamos un gráfico de calor utilizando la matriz de correlación 
    st.header('Análisis exploratorio de Madrid')
    st.subheader('Tabla de Correlación')
    with st.expander("**Correlaciones dentro de esta matriz**"):
        st.write("""
                El número de reseñas (0.694251) tiene una correlación positiva fuerte con reseñas por mes. 
                En cambio el número de reseñas (0.694251) tiene una correlación positiva fuerte con reseñas por mes
                """)
# ploteamos el gráfico de la matriz
    fig = px.imshow(corr, color_continuous_scale=px.colors.sequential.Jet)  
    fig.update_layout(width=800, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig)
    
# creamos un gráfico de pastel para ver la distribución en porcentajes  y lo añadimos al despegable para mejor visualización
with tab_plots:
    st.subheader('Distribución de los distritos en Madrid')
    with st.expander("**¿Qué porcentajes son los más reseñables?**"):
        st.write("""
                Centro es el distrito con más volumen dentro de Madrid (46.9%), 
                sin embargo Vicálvaro es el distrito con menos extensión con un 0.312%
                """)
        value_counts = df['neighbourhood_group'].value_counts()
        fig = go.Figure(data=[go.Pie(labels=value_counts.index, values=value_counts.values, hovertext=value_counts.index)])
        fig.update_layout(width=600, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig)
    neighbourhood_filter = st.selectbox("Selecciona el distrito para ver su ubicación", pd.unique(df["neighbourhood_group"]))
    # filtramos el dataframe según el valor seleccionado en el selectbox y añadimos el mapa
    df_filtered = df[df["neighbourhood_group"] == neighbourhood_filter]
    map = px.scatter_mapbox(df_filtered, lat='latitude', lon='longitude', color='neighbourhood_group', size_max=15, zoom=10, height=600)
    map.update_layout(mapbox_style='open-street-map')
    map.update_layout(margin={"r":80,"t":80,"l":80,"b":80})
    map.update_layout(width=800, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(map) 

# creamos un gráfico de barras para ver la distrución de los barrios dentro del expander y a su vez añadimos un mapa para su mejor visualización
with tab_plots:
    st.subheader('Distribución de los barrios en Madrid')
    with st.expander("**¿Cuántos barrios tiene madrid?**"):
        st.write("""
                Madrid está dividido administrativamente en 21 distritos, que a su vez se subdividen en 131 barrios, no necesariamente coincidentes con los barrios tradicionales.
                """)
        value_counts = df['neighbourhood'].value_counts()
        fig = go.Figure(data=[go.Bar(x=value_counts.index, y=value_counts.values, text=value_counts.index, textposition='outside')])
        fig.update_layout(xaxis_title='Barrios', yaxis_title='Frecuencia')
        fig.update_layout(width=600, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig)
    map = px.scatter_mapbox(df, lat='latitude', lon='longitude', color='neighbourhood',
                        size_max=15, zoom=10, height=600)
    map.update_layout(mapbox_style='open-street-map')
    map.update_layout(width=800, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    map.update_layout(margin={"r":80,"t":80,"l":80,"b":80}) 
    st.plotly_chart(map)

# creamos un enlace interactivo sobre el crimen de la comunidad de Madrid, le añadimos un expander para comentarios
with tab_plots:
    st.subheader('Crimen en la Comunidad de Madrid')
    with st.expander("**¿Es Madrid una ciudad insegura?**"):
        st.write("""
                La tasa de criminalidad en Madrid Centro puede ser más alta debido a una mayor densidad de población y turistas, lo que aumenta la probabilidad de que ocurran delitos.  
                Además, el centro de la ciudad puede ser un punto de atracción para individuos que buscan cometer delitos debido a la presencia de comercios y lugares de entretenimiento.
                No obstante la presencia de agentes de policía en el centro de Madrid es mayor, lo que induce a una mayor detección de delitos de manera precoz.
                """)
    st.write(f'<iframe src="https://www.datawrapper.de/_/qJFWw/" width="800" height="600" style="overflow:auto"></iframe>', unsafe_allow_html=True)
    url = "https://www.datawrapper.de/_/qJFWw/"
    st.markdown("[Datawrapper](%s)" % url)

# creamos una tabla de barras sobre los tipos de habitación de Madrid y añadimos un gráfico en forma de mapa mucho mas visual
with tab_plots:
    st.subheader('Tipos de habitación en Madrid')
    with st.expander("**Qué habitaciones nos podemos encontrar en Madrid**"):
        st.write("""
        1. Apartamentos: Estos apartamentos son la opción más elegida típicos de Madrid son una excelente opción para aquellos que buscan una estancia cómoda.
        2. Estudios: Al ser una elección más cara, es una opción menos recurrente
        3. Habitaciones en pisos compartidos: Al ser compartida, los viajeros eligen menos esta opción aunque sea una más económica.
                """)
        room_type_counts = df.room_type.value_counts()
        fig = px.bar(room_type_counts, x=room_type_counts.index, y=room_type_counts.values,
                color=room_type_counts.index,
                color_continuous_scale='Plasma',
                labels={'x':'', 'y':''})
        fig.update_layout(xaxis_title="Tipo de habitación", yaxis_title="Total")
        fig.update_layout(width=600, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig)
    map = px.scatter_mapbox(df, lat='latitude', lon='longitude', color='room_type',
                            size_max=15, zoom=10, height=600)
    map.update_layout(mapbox_style='open-street-map')
    map.update_layout(margin={"r":80,"t":80,"l":80,"b":80})  
    map.update_layout(width=800, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(map)

# hacemos un boxplot donde muestre la media de los datos de disponibilidad por barrio, añadimos un mapa con un seleccionable donde poderlo ver de manera más visual
with tab_plots:
    st.subheader('Distribución de la disponibilidad anual por barrio')
    with st.expander("**Disponibilidad de Airbnb en Madrid**"):
        st.write("""
                En Madrid la disponibilidad de los pisos es bastante alta, siendo en los pisos de la zona
                centro donde más nos costará encontrar una reserva disponible, dado que es la zona más turística.
                """)
        df_grouped = df.groupby(['neighbourhood_group'])['availability_365'].mean()
        fig = go.Figure()
        for neighbourhood_group, availability_365 in df_grouped.items():
            fig.add_trace(go.Box(y=df[df['neighbourhood_group'] == neighbourhood_group]['availability_365'], name=neighbourhood_group))
            fig.update_layout(xaxis_title='Barrios', yaxis_title='Disponibilidad anual')
            fig.update_layout(width=600, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig)
    # creamos la variable que contendrá la selección de barrios haciendo un df que solo contenta los barrios
    availability_filter = st.selectbox("Selecciona el barrio para ver la disponibilidad", pd.unique(df["neighbourhood"]))
    # filtramos el dataframe según el valor seleccionado en el selectbox
    df_filtered = df[df["neighbourhood"] == availability_filter]
    map = px.scatter_mapbox(df_filtered, lat='latitude', lon='longitude', color='availability_365',
                            size_max=15, zoom=10, height=600, color_continuous_scale='viridis')
    map.update_layout(mapbox_style='open-street-map')
    map.update_layout(margin={"r":80,"t":80,"l":80,"b":80})  
    map.update_layout(width=800, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(map)


#------------------------------------------------------------------------------------------CONCLUSIONES----------------------------------------------------------------------------------------------------------

# tercera pestaña
tab_plots = tabs[2]

# añadimos una imagen con unas conclusiones
with tab_plots:
    image = Image.open('C:\\Users\\rober\\Documents\\Rober\\Bootcamp\\Modulo 2\\20-Trabajo Módulo 2\\datos\\granviawide.png')
    st.header('Conclusiones')
    st.image(image, width=800)
    st.write('Skyline Gran Vía: Imagen creada con DALL-E-2')
    st.subheader('Conclusiones EDA Madrid')
    st.markdown('- **El número de propiedades en alquiler en Airbnb en Madrid ha aumentado significativamente en los últimos años.**')
    st.markdown('- **Un gran porcentaje de las propiedades en alquiler en Airbnb en Madrid son pisos completos, en lugar de habitaciones individuales.**')
    st.markdown('- **Muchas propiedades en alquiler en Airbnb en Madrid están registradas a través de anfitriones que tienen varias propiedades a la vez.**')
    st.markdown('- **Los precios de alquiler de las propiedades en Airbnb en Madrid varían ampliamente, pero en general son más altos en los barrios más turísticos de la ciudad.**')
    st.markdown('- **La mayoría de las propiedades en alquiler en Airbnb en Madrid están registradas en los barrios de Centro, Chamberí y Retiro.**')
    st.markdown('- **Madrid es una ciudad segura, pero si se busca una mayor tranquliad, una zona exterior a Madrid Centro puede ser buena elección**')

# hacemos un plot de un gráfico de palabras añadiendo un png de la bandera de Madrid
with tab_plots:
    st.subheader('Mapa de palabras')
    mask = np.array(Image.open('Flag_of_the_Community_of_Madrid.png'))
    wordcloud = WordCloud(mask=mask, background_color='white', contour_width=1, contour_color='red', min_font_size=5, collocations=False).generate(" ".join(df.neighbourhood))
    image_colors = ImageColorGenerator(mask)
    plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
    plt.axis('off')
    plt.savefig("madridflag.png", format="png")
    st.image(wordcloud.to_image(), caption='Wordcloud')

    # añadimos texto en html para despedir 
    st.markdown("<h1>Muchas Gracias</h1>", unsafe_allow_html=True)


#----------------------------------------------------------------------------------------------------END--------------------------------------------------------------------------------------------------------------