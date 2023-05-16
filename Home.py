import streamlit as st
from PIL import Image

st.set_page_config(layout="wide")

st.write("# Visualización de modelos de aprendizaje automático para la detección de COVID-19 en pacientes")

# Content
c1, c2 = st.columns(2)
c1.image(Image.open('images/python.png'))
c2.image(Image.open('images/sklearn.png'))
# c3.image(Image.open('images/polygon-logo.png'))

# st.sidebar.success("Select a demo above.")

st.markdown(
    """
    
    ## Propuesta de Diseño
    
    Rubén González Villanueva 
    
    Kevin Quesada Montero
    
    
    ### Descripción
    Por medio de esta aplicación es posible analizar de forma detallada el impacto de las diferentes variables en la predicción de distintos modelos de aprendizaje automático.
    
    ### Fuentes:
    - Datos: (https://www.kaggle.com/datasets/meirnizri/covid19-dataset?datasetId=2633044&sortBy=relevance)
    - https://streamlit.io/
    - https://shap.readthedocs.io/
    
    """
)

c1, c2, c3 = st.columns(3)
# with c1:
#     st.info('**Data Analyst: [@AliTslm](https://twitter.com/AliTslm)**', icon="💡")
# with c2:
#     st.info('**GitHub: [@alitaslimi](https://github.com/alitaslimi)**', icon="💻")
# with c3:
#     st.info('**Data: [Flipside Crypto](https://flipsidecrypto.xyz)**', icon="🧠")
with c1:
    st.info('**Data Analyst: XX XX**', icon="💡")
with c2:
    st.info('**GitHub: XX XX**', icon="💻")
with c3:
    st.info('**Data: XX XX**', icon="🧠")