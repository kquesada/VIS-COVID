import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.metrics import ConfusionMatrixDisplay
from xgboost import XGBClassifier, XGBRFClassifier #
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)

import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import shap

import matplotlib
# matplotlib.use('TkAgg')
st.set_option('deprecation.showPyplotGlobalUse', False)


st.set_page_config(layout="wide")

# st.set_page_config(page_title="sss", page_icon="")

# Set up the left-side panel
st.sidebar.title('Select Options')
# Create a file uploader widget

def preprocess_data(data, categorical_features, continuous_features, target_col = 'CLASIFFICATION_FINAL'):
    X = data.drop(columns=[target_col])
    y = data[target_col]

#     X[categorical_features] = X[categorical_features].apply(
#         LabelEncoder().fit_transform
#     )
#     X[continuous_features] = RobustScaler().fit_transform(X[continuous_features])

    return X, y

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

categorical_features = [
    "USMER",
    "MEDICAL_UNIT",
    "SEX",
    "PNEUMONIA",
    "PREGNANT",
    "DIABETES",
    "COPD",
    "ASTHMA",
    "INMSUPR",
    "HIPERTENSION",
    "OTHER_DISEASE",
    "CARDIOVASCULAR",
    "OBESITY",
    "RENAL_CHRONIC",
    "TOBACCO",
]
continuous_features = ["AGE"]

# 2 SELECT MODEL
pretrained_models = ['LogisticRegression', 'XGBRFClassifier', 'XGBClassifier']
model_selector = st.sidebar.selectbox('1 - Select Pre-trained Model', pretrained_models)

# 3 SELECT PREDICTOR
if 'data' not in st.session_state:
    st.write('# Please load a dataset')

if 'data' in st.session_state:# is not None:
    st.write(" # Loaded Data")
    st.write(st.session_state['data'].drop(columns=['CLASIFFICATION_FINAL']))
        
    # st.write(p_data)
    selected_indices = st.sidebar.multiselect('2 - Select one or more subjects:', st.session_state['data'].index, max_selections=10)
    if len(selected_indices) > 0:
        selected_rows = st.session_state['data'].loc[selected_indices]
        st.write('# Selected Subject Data')
        st.write(selected_rows.drop(columns=['CLASIFFICATION_FINAL']))
        if not selected_rows.empty:
            # data preprocessing
            p_data = preprocess_data(selected_rows, categorical_features, continuous_features)

# 5 INFERENCE PROCESS
if len(selected_indices) > 0:
    inference = st.sidebar.button('3 - Inference')

    models = {
        'LogisticRegression': LogisticRegression(),
        'XGBClassifier': XGBClassifier(),
        'XGBRFClassifier': XGBRFClassifier()
    }
    metadata_models = {
        "LogisticRegression": "Selected model: LogisticRegression\n\nSize: 1.1kB\n\nOverall Metrics:\n\nPrecision :  0.628186148300721\n\nRecall :  0.25096430775560585\n\nAccuracy :  0.6595149026244812\n\nF1 Score :  0.3586469204762605",
        "XGBClassifier": "Selected model: XGBClassifier\n\nSize: 396kB\n\nOverall Metrics:\n\nPrecision :  0.636918291661682\n\nRecall :  0.2738119728450936\n\nAccuracy :  0.6653189029951568\n\nF1 Score :  0.3829802539294321",
        "XGBRFClassifier": "Selected model: XGBRFClassifier\n\nSize: 501kB\n\n\nOverall Metrics:\n\nPrecision :  0.6247047104027323\n\nRecall :  0.28220787903723515\nAccuracy :  0.663402119679463\n\nF1 Score :  0.38878408658146685"
    }
    outcome_val = {
        '0': 'COVID Negative',
        '1': 'COVID Positive'
    }

    if inference:
        model = models[model_selector]
        model_metadata = metadata_models[model_selector]
        # load shap
        filename_explainer = f"shaps/{type(model).__name__}-explainer.h5"
        explainer = pickle.load(open(filename_explainer, "rb"))
        # load model
        filename_model = f"models/{type(model).__name__}.h5"
        loaded_model = pickle.load(open(filename_model, "rb"))
        st.divider()
        st.write('# Model metadata')

        col1, col2 = st.columns([1, 1])
        col1.subheader("Metrics")
        col1.write(model_metadata)

        col2.subheader("Details")
        col2.write(loaded_model)
        st.divider()

        # predictions
        if len(selected_indices) == 1:
            pred = loaded_model.predict(p_data[0])[0]
        if len(selected_indices) > 1:
            pred = loaded_model.predict(p_data[0])
            d = {'Subject': selected_indices, 'Pred': pred}
            pred_df = pd.DataFrame(data=d)

            pred_df.loc[pred_df['Pred'] == 0, 'Prediction'] = 'COVID Negative'  
            pred_df.loc[pred_df['Pred'] == 1, 'Prediction'] = 'COVID Positive' 

            pred_message = ''
            for p in pred_df.index:
                value = pred_df.loc[p]
                subject_value = value['Subject']
                pred_value = value['Pred']
                prediction_value = value['Prediction']

                string = f'Subject {subject_value} - outcome :blue[{pred_value}] or :blue[{prediction_value}]\n\n'
                pred_message =  pred_message + string 

        shap_values = explainer(p_data[0])

        if len(selected_indices) == 1:
            st.subheader(f"Predicted Value:")
            st.write(f'Subject {selected_indices[0]} - outcome :blue[{str(pred)}] or :blue[{outcome_val[str(pred)]}]')
            st.divider()
            waterfall = shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(waterfall)

            fig = plt.figure(figsize=(10,15))
            ax1 = fig.add_subplot(121) #221
            shap.plots.bar(shap_values, show=False)
            ax1.title.set_text('Bar')

            ax2 = fig.add_subplot(122) #222
            shap.plots.beeswarm(shap_values, show=False)
            ax2.title.set_text('Beeswarm')

            plt.tight_layout()
            st.pyplot(plt)

            shap.initjs()
            st_shap(shap.plots.force(shap_values[0:100]), 400)

        if len(selected_indices) > 1:
            st.subheader(f"Predicted Values:")
            # st.write(pred_df)
            st.write(pred_message)
            st.divider()
            beeswarm = shap.plots.beeswarm(shap_values, show=False)
            st.pyplot(beeswarm)

            shap.initjs()
            st_shap(shap.plots.force(shap_values[0:100]), 400) 

            heat = shap.plots.heatmap(shap_values, max_display=20)
            st.pyplot(heat)

    

