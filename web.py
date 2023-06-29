#!pip install streamlit -q
#!pip install streamlit-lottie
#!pip install Pillow



import streamlit as st
import requests
from streamlit_lottie import st_lottie
from PIL import Image


#Funcion para nuestra animacion
def load_lottieurl(url):
  r = requests.get(url)
  if r.status_code != 200:
    return None
  return r.json()

lottie_coding = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_9wpyhdzo.json")
imagen_video = Image.open("C:/Users/adiaz/Documents/stregresion/adiaz.jpg") 


with st.container():
  st.subheader("Hola bienvenido a mi sitio web")
  st.title("Introduccion a la ciencia de datos")
  st.write("Bienvenido a mi canal :wave:")
  st.write("[Mas informacion >](https://www.youtube.com/watch?v=xzXunskiNcg)")

with st.container():
  st.write("---")
  left_column, right_column = st.columns(2)
  with left_column:
    st.header("Mi objetivo")
    st.write(
      """
        Texto aquí
      """
    )
    st.write("[Youtube >](https://www.youtube.com/watch?v=drOP5OupLFA)")
  with right_column:
    st_lottie(lottie_coding, height=300, key="coding")

with st.container():
  st.write("--")
  st.header("Mis videos")
  image_column, text_column = st.columns((1, 2))
  with image_column:
    st.image(imagen_video)
  with text_column:
    st.write(
      """
      texto aquí
      """
    )
    st.markdown("[Ver video...](https://www.youtube.com/watch?v=7DcrJh1ZsFo)")

  