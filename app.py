import streamlit as st 
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd 
import numpy as np
import pickle
from datetime import datetime
from src.model import algorithm
from src.utils import transform_, plot_

model = pickle.load(open(r'./Artifacts/model.bin', "rb"))
features = pickle.load(open(r'./Artifacts/features.bin', "rb"))
vectorizer = pickle.load(open(r'./Artifacts/vectorizer.bin', "rb"))


st.title("Boston House Prediction")

st.sidebar.write(f'Last edited on {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
add_selectbox = st.sidebar.selectbox(
    'What do you want to do?',
    ('Summary', 'Prediction')
)

if add_selectbox == 'Prediction':
    col1, col2, col3 = st.columns(3)

    crim_ = col1.number_input(
        "Input Crime Rate (crim)"
    )

    zn_ = col1.number_input(
        "Input Proportion of residential land zoned for lots over 25,000 sqft (zn)"
    )

    indus_ = col1.number_input(
        "Input Proportion of non-retail business acres per town (indus)"
    )

    chas_ = col1.selectbox(
        "Around Charles River? (chas)",
        ("Yes", "No")
    )
    if chas_ == 'Yes':
        chas_ = 1
    else:
        chas_ = 0

    nox_ = col2.number_input(
        "Input nitric oxides concentration (nox)"
    )

    rm_ = col2.number_input(
        "Input average number of rooms per dwelling (rm)"
    )

    age_ = col2.number_input(
        "Input proportion of owner-occupied units built prior to 1940 (age)"
    )

    dis_ = col2.number_input(
        "Input weighted distances to five Boston employment centers"
    )

    rad_ = col3.number_input(
        "Input index of accessibility to radial highways (rad)"
    )

    tax_ = col3.number_input(
        "Input full-value property-tax rate per $10,000"
    )

    ptratio_ = col3.number_input(
        "Input pupil-teacher ratio by town"
    )

    b_ = col3.number_input(
        "Input proportion of blacks by town"
    )

    lstat_ =col1.number_input(
        "Input lower status of the population"
    )

    X = np.array([
        crim_, zn_, indus_, chas_, nox_,
        rm_, age_, dis_, rad_, tax_,
        ptratio_, b_, lstat_
    ]).reshape(1,-1)

    Transform = st.selectbox(
        "Transform Data", ("Yes", "No")
    )
    if Transform == "Yes":
        X = vectorizer.transform(pd.DataFrame(X).to_dict(orient = 'records'))
    elif Transform == "No":
        X = X
    
    pred_ = "None"
    if st.button(
        "Get Result"
    ):
        pred_ = model.predict(X)
    
    if pred_ != "None":
        st.success(f"Estimated Price = ${pred_[0]}")

elif add_selectbox == 'Summary':
    st.sidebar.header("Upload File")
    file_ = st.sidebar.selectbox (
        "Data Source",
        ("Upload", "Default Data")
    )
    if file_ == "Upload":
        uploaded_file = st.sidebar.file_uploader("Choose a file!")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, keep_default_na = False)
    elif file_ == "Default Data": 
        df = pd.read_csv('./Data/boston_housing.csv', keep_default_na = False)
    try:
        st.header("Raw Data")
        st.write(df.head())

        used_ = features
        used_.remove('chas')

        select_cols = st.selectbox(
        'Select Columns',
        tuple(used_)
        )

        temp = df.corr()['medv'].drop('medv')
        hehe = pd.DataFrame()
        hehe['cols'] = temp.index
        hehe['corr'] = temp.values
        hehe['corr abs'] = abs(temp.values)
        hehe = hehe.sort_values(by = 'corr abs', ascending = False).reset_index(drop = True)
        corr_ = hehe.query(f"cols == '{select_cols}'")['corr'].values[0]
        idx_ = hehe.query(f"cols == '{select_cols}'")['corr'].index[0]

        st.metric(
            f'Correlation to House Price',
            idx_ + 1,
            round(corr_,3)
            )

        fig_def = px.scatter(
            df,
            x = select_cols,
            y = 'medv',
            color = 'chas',
            hover_name = 'chas',
            log_x = True,
        )
        st.plotly_chart(
            fig_def, theme = 'streamlit',
            use_container_width= True
        )

        tab1, tab2 = st.tabs(["Original Data", "Transformed Data"])
        with tab1:
            bin_slider = st.slider(
            "Bin Size", 
            min_value = 0.0, max_value = 1.0,step = 0.01
            )
            fig_dist = ff.create_distplot(
                [df[select_cols]],
                group_labels= [select_cols],
                bin_size=[bin_slider]
            )

            fig_box = px.box(
                df, x = select_cols
            )
            st.markdown(f'{select_cols} Distribution plot')
            st.plotly_chart(fig_dist, use_container_width=True)
            st.markdown(f'{select_cols} Boxplot')
            st.plotly_chart(fig_box, theme = 'streamlit',use_container_width= True)

        with tab2:
            df_transformed = df.copy()
            for i in df_transformed.columns:
                df_transformed[i] = transform_(df_transformed, i)

            bin_slider_tf = st.slider(
            "Bin Size for Transformed Data",
            min_value = 0.0, max_value = 1.0,step = 0.01
            )
            fig_dist = ff.create_distplot(
                [df_transformed[select_cols]],
                group_labels= [select_cols],
                bin_size= [bin_slider_tf]
            )

            fig_box = px.box(
                df_transformed, x = select_cols
            )
            st.markdown(f'{select_cols} Distribution plot')
            st.plotly_chart(fig_dist, use_container_width=True)
            st.markdown(f'{select_cols} Boxplot')
            st.plotly_chart(fig_box, theme = 'streamlit',use_container_width= True)
    except:
        st.write("Input Data First!")




