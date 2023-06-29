import streamlit as st


#importar las bibliotecas tradicionales de numpy y pandas
import numpy as np
import pandas as pd

#importar las biliotecas graficas e imágenes
import plotly.express as px


from PIL import Image
import urllib.request
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

import requests


urllib.request.urlretrieve('https://github.com/adiacla/precipitacion/blob/main/adiaz.jpg?raw=true',"adiaz.jpg")
imagen_video = Image.open("adiaz.jpg") 



#Librerias no usadas
#from streamlit_lottie import st_lottie
#import requests

## Iniciar barra lateral en la página web y título e ícono

st.set_page_config(
  page_title="Analisis de lluvia",
  page_icon=("https://github.com/adiacla/precipitacion/blob/main/rain.ico"),
  initial_sidebar_state='auto'
  )


#Primer contenedor

with st.container():
  st.subheader("Análisis exploratorio de la serie de tiempo de precipitación")
  st.title("Implementación en Python")
  st.write("Realizado por Alfredo Díaz Claros:wave:")
  st.image(imagen_video , width=100)
  st.write("""
**Introducción** Los datos fueron tomados de https://www.datos.gov.co/Ambiente-y-Desarrollo-Sostenible/Precipitaciones/ksew-j3zj,  que tiene las siguientes clausulas: 
1.Los datos a visualizar o descargar a continuación no han sido validados por el **IDEAM**. 
2. Los datos aquí dispuestos son datos crudos instantáneos provenientes de los sensores de las estaciones automáticas de la red propia y/o producto de convenios interadministrativos con terceras entidades. 
3. Los datos son puestos a disposición de la ciudadanía como mecanismo de transparencia y para proveer una herramienta de apoyo a la gestión del riesgo territorial y como datos abiertos en cumplimiento de la Ley 1712 de 2014. 
4. Es posible que los datos aquí dispuestos tengan cierto retraso en el tiempo debido a las frecuencias de envío de datos de los sensores y los medios de transmisión utilizados. 
5. Los datos presenten errores y/o inconsistencias estando incluso por fuera de los límites considerados normales, producto de fallas en los sensores de origen. 
6. El posterior uso e interpretación que se le dé para cualquier finalidad queda bajo la exclusiva responsabilidad del portador de los datos. 
7. Cualquier destino que se le dé a los datos exime al **IDEAM** de realizarles cualquier tipo de justificación o iniciarles proceso alguno de validación posterior, cuya fuente primaria deben ser los canales oficialmente dispuestos por la Entidad para el suministro de información oficial y validada. 
8. Por las razones expuestas anteriormente los datos dispuestos no podrán ser utilizados como evidencia jurídica ante entes de control acerca de la ocurrencia o no de fenómenos hidroclimatológicos o de soporte a cualquier tipo de situación o evento ocurrido como consecuencia de estos.

""")

url = 'https://drive.google.com/file/d/11F9XsCOnaxl633mFKg1ZItY7ml_XrGXI/view?usp=sharing'
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]

datos = pd.read_csv(path,delimiter=",")
datos['FechaObservacion']=pd.to_datetime(datos['FechaObservacion'])
datos['Ano'] = datos['FechaObservacion'].dt.year 
datos['Mes'] = datos['FechaObservacion'].dt.month 
 


with st.container():
  st.write("---")
  left_column, right_column = st.columns(2)
  with left_column:
    st.subheader("Importar dataset y detalles")
    st.write(
      """
      El objetivo de este trabajo es generar una herramienta en código Python para analizar los datos históricos de precipitaciones de la base de datos de las distintas estaciones meteorológicas del IDEAM.
      Para el análisis de  los patrones de precipitación y sus tendencias usaremos  ***Python*** y mostraremos los diferentes métodos.      
      Iniciamos cargando las bibliotecas y los datos, y transformamos la columna en tipo fecha.
      """
    )

  with right_column:
      st.subheader("Código")
      code = '''
      import requests
      from PIL import Image
      import matplotlib.pyplot as plt
      import numpy as np
      import pandas as pd
      import plotly.express as px
      from datetime import datetime
      
      datos = pd.read_csv('C:/Users/adiaz/Documents/stregresion/Precipitacion.csv', delimiter=",")
      datos['FechaObservacion']=pd.to_datetime(datos['FechaObservacion'])
      
      datos['Ano'] = datos['FechaObservacion'].dt.year 
      datos['Mes'] = datos['FechaObservacion'].dt.month 
  
      datos.info()
 
     '''
      st.code(code, language="python", line_numbers=False)

st.subheader("Detalle del dataset")

st.write("El número de registros cargados es: ", len(datos), "comprendido desde ", str(datos['FechaObservacion'].min()), " hasta ", datos['FechaObservacion'].max() )
st.write("El número de departamentos es ", len(datos['Departamento'].unique()), ", de", len(datos['Municipio'].unique()), " municipos",  " con ",len(datos['CodigoEstacion'].unique()),  " estaciones")
st.write( datos.info())
st.write(datos.head(5))
todos=a = np.array(["Todos"])


#Opciones de la barra lateral

urllib.request.urlretrieve('https://github.com/adiacla/precipitacion/blob/main/rain_3.png?raw=true',"rain_3.png")
logo= Image.open("rain_3.png") 
st.sidebar.write('...')
st.sidebar.image(logo, width=100)
st.sidebar.header('Opciones de análisis')


def seleccionar(datos):
  #Filtrar por departamento
  st.sidebar.subheader('Selector de ubicación')
  df=datos.copy()
  lista_departamentos=df.Departamento.unique()
  lista_departamentos=np.sort(lista_departamentos)
  lista_departamentos=np.append(todos, lista_departamentos)
  selecDepartamento=st.sidebar.selectbox("Seleccione el Departamento",lista_departamentos)
  if selecDepartamento!="Todos":
    df=df[df['Departamento']==selecDepartamento]

  #Filtrar por municipio
  lista_municipios=df.Municipio.unique()
  lista_municipios=np.sort(lista_municipios)
  lista_municipios=np.append(todos,lista_municipios)
  selecMunicipio=st.sidebar.selectbox("Seleccione el municipio",lista_municipios)
  if selecMunicipio!="Todos":
    df=df[df["Municipio"]==selecMunicipio]

  #Filtrar por estaciones
  lista_estaciones=df.NombreEstacion.unique()
  lista_estaciones=np.sort(lista_estaciones)
  lista_estaciones=np.append(todos,lista_estaciones)
  selecEstacion=st.sidebar.selectbox("Selecciones la estacion",lista_estaciones)
  if selecEstacion!="Todos":
    df=df[df["NombreEstacion"]==selecEstacion]

  st.sidebar.subheader('Selector de fecha')
  
  lista_año=df['Ano'].unique()
  lista_año=np.sort(lista_año)
  lista_año=np.append(todos,lista_año)
  selecano=st.sidebar.selectbox("Seleccione el Año",lista_año)
  if selecano!="Todos":
    df=df[df["Ano"]==int(selecano)]

  
  lista_mes=df['Mes'].unique()
  lista_mes=np.sort(lista_mes)
  lista_mes=np.append(todos,lista_mes)
  selecmes=st.sidebar.selectbox("Selecciones el mes",lista_mes)
  if selecmes!="Todos":
    df=df[df["Mes"]==int(selecmes)]

  return selecDepartamento,selecMunicipio,selecEstacion, selecano,selecmes,df

selecDepartamento,selecMunicipio,selecEstacion,selecano,selecmes,df=seleccionar(datos)

st.write("Se ha filtrado por los siguientes criterios:")
st.write("Departamento : ", selecDepartamento)
st.write("Municipio : ", selecMunicipio)
st.write("Estación :", selecEstacion)
st.write("del año", selecano)
st.write("del mes",selecmes)

  
st.write("Total de datos del dataset", len(df))
st.write(df.head(5))


#st.write(datos.describe())
with st.container():
  st.subheader("Análisis gráfico ")
  st.title("Serie de tiempo de las precipitaciones en mm timestamp")
  st.write("""
           Precipitación(mm de agua) reportada automaticamente por los sensores inalambricos de las estaciones meteorologicas del IDEAM, en función del tiempo.
           """)
  # Create figure and plot space
  fig, ax = plt.subplots(figsize=(10, 10))
  # # Add x-axis and y-axis
  ax.scatter(df['FechaObservacion'],
             df['ValorObservado'],
             color='purple')

# Set title and labels for axes
  ax.set(xlabel="Date",
        ylabel="Precipitación (mm)",
        title="Precipitación en el tiempo")

  st.pyplot(fig)
  
  st.write("""
            Explicación de la gráfica en función del tiempo.
            """)


with st.container():
  st.subheader("Gráfica de precipiación agrupada por horas")
  st.title("Serie de tiempo de las precipitaciones en mm y en horas (promedio)")
  st.write("""
           Para simplificar el el análisis que tiene muchos puntos de datos debido a los registros por cada timestamp, se pueden agregar los datos para cada hora usando el método.
           Se tomará la media de las observaciones.
           """)
  
  dfh=df.loc[:, ['FechaObservacion', 'ValorObservado']]
  dfh = dfh.groupby('FechaObservacion').mean()
  
  st.write(dfh.head())
  
  # Create figure and plot space
  fig, ax = plt.subplots(figsize=(10, 10))
  # Add x-axis and y-axi
  ax.plot(dfh.index.values,
          dfh['ValorObservado'],
          color='red')

  # Set title and labels for axes
  ax.set(xlabel="Fecha",
       ylabel="Precipitacion (mm)",
       title="Precipitacion promedio por hora")
  st.pyplot(fig)
  
  st.write("""
            Explicación de la gráfica agrupada en hora
            """)

  
with st.container():
  st.subheader("Gráfica de precipiación agrupada por dia")
  st.title("Análsis diario de precipitaciones")
  st.write(""""        
           Para agregar o muestrear temporalmente los datos durante un período de tiempo, puede tomar todos los valores para cada día y resumirlos.
           
           En este caso, desea una precipitación diaria total, por lo que utilizará el resample() método junto con .sum(). 
                     
           En el paso anterior dfh (por horas) crea el indice tipo fechas, podemos usar el resample.
           """)
  dfd = dfh.resample('D').sum()
  st.write(dfd.head())
  
  # Create figure and plot space
  fig, ax = plt.subplots(figsize=(10, 10))
  # Add x-axis and y-axi
  ax.plot(dfd.index.values,dfd['ValorObservado'],color='red')

  # Set title and labels for axes
  ax.set(xlabel="Fecha",
       ylabel="Precipitation (mm)",
       title="Diaria Precipitation")
  st.pyplot(fig)
  
  
with st.container():
  st.subheader("Gráfica de precipiación agrupada por mes")
  st.title("Análsis mensualde precipitaciones")
  st.write("""
             
           En este caso, desea una precipitación mensual total, por lo que utilizará el resample() método junto con .sum(). 
           """)
  dfm = dfh.resample('M').sum()
  st.write(dfm)
  
  # Create figure and plot space
  fig, ax = plt.subplots()
  # Add x-axis and y-axi
  ax.bar(dfm.index.values,
          dfm['ValorObservado'],
          color='green')

  # Set title and labels for axes
  ax.set(xlabel="Fecha",
       ylabel="Precipitation (mm)",
       title="Precipitacion Mensual")
  st.pyplot(fig)
  
  ###### por unbicación
  
  
with st.container():
  st.subheader("Análisis  de precipiación geolocalizadas")
  st.title("Análisis total de precipitaciones por ubicación")
  st.write("""
             
           Utilizaremos ahora el análisis de por geolocalización para las precipitaciones. 
           """)
  dfc=df.loc[:, ['FechaObservacion', 'CodigoEstacion','ValorObservado','Latitud','Longitud','Municipio']]
  dfc = dfc.groupby(['CodigoEstacion', 'Latitud','Longitud','Municipio']).ValorObservado.sum()
  dfc=dfc.reset_index(['CodigoEstacion', 'Latitud','Longitud','Municipio'])
  media=dfc['ValorObservado'].mean()
  desviacion=dfc['ValorObservado'].std()
  limite=media+2*desviacion
  st.write("La Media de lluvia es: ",media)
  dfc['Diferencia media']=media-dfc['ValorObservado']
  dfc['Diferencia 2 Desviaciones']=limite-dfc['ValorObservado']
  
  
  def negativos(col):
    highlight = 'color: green;'
    default = 'color: red;'
    return [highlight if e > 0 else default for e in col]  
  
  dfc_style = dfc.style.apply(negativos, axis=0, subset=['Diferencia media','Diferencia 2 Desviaciones'])
  st.dataframe(dfc_style)
  
  # Create figure and plot space
  fig, ax = plt.subplots()
  # Add x-axis and y-axi
  ax.scatter(dfc.Longitud,
             dfc.Latitud,
             marker="o", 
             s=dfc.ValorObservado,
             color='blue')

  # Set title and labels for axes
  ax.set(xlabel="Longitud",
       ylabel="Latitud",
       title="Precipitación total")
  st.pyplot(fig)
  
with st.container():
    # Create figure and plot space
  fig = px.scatter_mapbox(dfc, lat = 'Latitud', lon = 'Longitud', size = 'ValorObservado',
                          color='ValorObservado',
                          hover_name="Municipio",
                          center = dict(lat = 7.12, lon =-73.14 ),
                          color_continuous_scale=px.colors.cyclical.IceFire,
                          zoom = 6, mapbox_style = 'open-street-map')
  st.plotly_chart(fig, use_container_width=True)
   
with st.container():
    # Create figure and plot space
  fig = px.density_mapbox(dfc, lat = 'Latitud', lon = 'Longitud', z = 'ValorObservado',
                          hover_name='Municipio',
                          center = dict(lat = 7.12, lon =-73.14 ),
                          zoom = 6,
                          mapbox_style = 'open-street-map',
                          color_continuous_scale = 'rainbow')
  st.plotly_chart(fig, use_container_width=True)
  
with st.container():
  st.subheader("Detección de Anomalias")
  st.title("Análisis de precipitaciones")
  st.write("""
           Los Outliers pueden significar varias cosas:
           
           **ERROR**: Si tenemos un grupo de “edades de personas” y tenemos una persona con 160 años, seguramente sea un error de carga de datos. En este caso, la detección de outliers nos ayuda a detectar errores.
           **LIMITES**: En otros casos, podemos tener valores que se escapan del “grupo medio”, pero queremos mantener el dato modificado, para que no perjudique al aprendizaje del modelo de ML.
           **Punto de Interés**: puede que sean los casos “anómalos” los que queremos detectar y que sean nuestro objetivo (y no nuestro enemigo!)
           """)
  
  nmp=dfd['ValorObservado'].to_numpy()
  valor_unique, counts = np.unique(nmp, return_counts=True)
  sizes = counts*100
  colors = ['blue']*len(valor_unique)
  rango=range(int(len(valor_unique)/4))
  for i in rango:
    colors[-i] = 'red'
  fig, ax = plt.subplots()
  ax.axhline(1, color='k', linestyle='--')
  ax.scatter(valor_unique, np.ones(len(valor_unique)), s=sizes, color=colors)
  ax.set(xlabel="Datos mensuales",
         title="Valores Atipicos")
  st.pyplot(fig)

with st.container():
  st.subheader("Detección de Anomalias")
  st.title("Análisis de Outlier")
  st.write("""
           Los Outliers pueden significar varias cosas:
           **ERROR**: Si tenemos un grupo de “edades de personas” y tenemos una persona con 160 años, seguramente sea un error de carga de datos. En este caso, la detección de outliers nos ayuda a detectar errores.
           
           **LIMITES**: En otros casos, podemos tener valores que se escapan del “grupo medio”, pero queremos mantener el dato modificado, para que no perjudique al aprendizaje del modelo de ML.
           
           **Punto de Interés**: puede que sean los casos “anómalos” los que queremos detectar y que sean nuestro objetivo (y no nuestro enemigo!)
           """)
  fig = px.box(y = dfd['ValorObservado'], color_discrete_sequence = ['green'])
  st.plotly_chart(fig, use_container_width=True)
  
  st.write(dfd['ValorObservado'].describe())
  q1=dfd['ValorObservado'].quantile(0.25)
  q3=dfd['ValorObservado'].quantile(0.75)
  IQR=q3-q1
  limite_sup=q3+1.5*IQR
  limite_inf=q1-1.5*IQR
  outliers = dfd[((dfd['ValorObservado']<(q1-1.5*IQR)) | (dfd['ValorObservado']>(q3+1.5*IQR)))]
  st.write("Los Outliers son ")
  st.write(outliers)
  
  st.subheader("Detección de Anomalias")
  st.title("Análisis histograma")
  st.write("""
           Histograma de frecuencia
           """)
  st.write(len(dfd['ValorObservado']))
  fig = px.histogram(dfd, x='ValorObservado')
  st.plotly_chart(fig)
  
  fig, ax = plt.subplots()
  ax.plot(dfd['ValorObservado'], marker = "o")
  ax.set_title('Comportamiento diario')
  st.pyplot(fig)


st.sidebar.subheader('Parámetros de Sigma')
wind = st.sidebar.slider('Ventana para rolling-Promedio movil?', 5, 30, 10)
sigma = st.sidebar.slider('Parámetro sigma?', 2, 3, 2)

with st.container():
  st.subheader("Suavizado usando promedio móvil")
  st.title("Función de promedio móvil")
  st.write("""
           Es una técnica ingenua y efectiva en el pronóstico de series de tiempo. El suavizado es una técnica aplicada a series de tiempo para eliminar la variación de grano fino entre los pasos de tiempo.

El suavisamiento se usa para eliminar el ruido y exponer mejor la señal de los procesos causales subyacentes. Los promedios móviles son un tipo de suavizado simple y común utilizado en el análisis de series temporales y el pronóstico de series temporales.

La función (rolling ) en el objeto Series Pandas agrupará automáticamente las observaciones en una ventana. Puede especificar el tamaño de la ventana y, por defecto, se crea una ventana de seguimiento. Una vez que se crea la ventana, podemos tomar el valor medio, y este es nuestro conjunto de datos transformado.

A continuación la transformación del conjunto de datos de lluvias en un promedio móvil con un tamaño de ventana.
           """)
  dfd["movil"]=dfd['ValorObservado'].rolling(window=wind).mean()
  fig, ax = plt.subplots()
  ax.plot(dfd['movil'])
  ax.set_title('Promedio móvil')
  st.pyplot(fig)

with st.container():
  st.subheader("Descomposición aditiva")
  st.title("Función de descompositición estacional")
  st.write("""
Algunos patrones distinguibles aparecen cuando trazamos los datos. Si la serie temporal tiene un patrón de estacionalidad, (Aunque en precicipiaciones son muy bajas o focalizadas por las estaciones en Colombia como en invierno y verano en zona tropical)

Aunque no tenemos en los datos identificadas las estaciones, vamos visualizar nuestros datos utilizando un método llamado Descomposición de series de tiempo que nos permite descomponer nuestras series de tiempo en tres componentes distintos: tendencia, estacionalidad y ruido.
           """)
  estacional= seasonal_decompose(dfd['ValorObservado'], model="additive",extrapolate_trend='freq')
  st.write(estacional)
  fig, ax = plt.subplots()
  fig=estacional.plot()
  st.pyplot(fig)
  
  st.write("""
           Como podemos observar en los gráficos que realizamos anteriormente, el comportamiento de la serie de tiempo con la que estamos trabajando parece ser totalmente aleatorio y las medidas móviles que calculamos tampoco parecen ser de mucha utilidad para acercar la serie a un comportamiento estacionario.
           """)
with st.container():
  st.subheader("Detección de Anomalias")
  st.title("Método 2 Sigma")
  st.write("""
           Los Outliers 
           """)
  

  dfd["Techo del Roling"] = dfd['ValorObservado'].rolling(window=wind).mean() + (sigma * dfd['ValorObservado'].rolling(window=wind).std())
  dfd["Anomalias"] = dfd.apply(lambda row: row['ValorObservado'] if (row['ValorObservado']>=row["Techo del Roling"]) else 0
  , axis=1)
  fig, ax = plt.subplots()
  ax.plot(dfd['ValorObservado'], marker = "o")
  ax.plot(dfd["Techo del Roling"])
  ax.plot(dfd["Anomalias"])
  ax.set_title('Comportamiento diario')
  st.pyplot(fig)
  st.write("Tabla de lluvias y anomalías")
  
  def text_format(val):
    color = 'salmon' if val < 0 else 'white'
    return 'background-color: %s' % color
  
  dfd_style=dfd.style.background_gradient(subset=["Anomalias"], cmap="Reds")
  st.dataframe(dfd_style)
  #dfd_style=dfd.style.applymap(text_format)
  #st.dataframe(dfd_style)
  
  df_pivot = pd.DataFrame(df.groupby(['Municipio','Mes','Ano']).ValorObservado.sum())
  m = {"1":'Enero',
        "2":'Febrero',
        "3":'Marzo',
        "4":'Abril',
        "25":'Mayo',
        "6":'Junio',
        "7":'Julio',
        "8":'Agosto',
        "9":'Septiembre',
        "10":'Octubre',
        "11":'Noviembre',
        "12":'Diciembre'}
  df_pivot.replace(m, inplace=True)
  st.write(df_pivot)
  
  df_pivot=df_pivot.pivot_table(index=['Mes'], columns=['Municipio'], values='ValorObservado').reset_index()
  st.write(df_pivot)
  label=df_pivot.columns.tolist()
  fig, ax = plt.subplots()
  for index in range(df_pivot.shape[1]):
    if index!=0:
      ax.plot(df_pivot['Mes'],df_pivot.iloc[: , index].values, marker = "o", label = label[index])
  ax.legend(loc ='upper left',fontsize="4")
  ax.set_title('Lluvias por mes')
  st.pyplot(fig)
  
  