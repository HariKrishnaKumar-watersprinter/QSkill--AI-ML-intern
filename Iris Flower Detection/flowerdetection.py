import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler,LabelEncoder,PolynomialFeatures
from lightgbm import LGBMClassifier
import seaborn as sns
from scipy import stats
import streamlit as st
@st.cache_resource
def train_model():
    df = sns.load_dataset('iris')
    df.drop_duplicates(inplace=True)
    species_unique = df['species'].unique()
    color_map = {species: idx for idx, species in enumerate(species_unique)}
    df['species_color'] = df['species'].map(color_map)
    
    def cap_outliers_iqr(df, column, iqr_factor=1.5):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
    
        lower_bound = Q1 - iqr_factor * IQR
        upper_bound = Q3 + iqr_factor * IQR
    
    
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        return df

    
    le = LabelEncoder()
    df['species'] = le.fit_transform(df['species'])
    df['sepal_ratio'] = df['sepal_length'] / (df['sepal_width'] + 1e-6)
    df['petal_ratio'] = df['petal_length'] / (df['petal_width'] + 1e-6)
    df['sepal_area'] = df['sepal_length'] * df['sepal_width']
    df['petal_area'] = df['petal_length'] * df['petal_width']
    numerical_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    poly_features = poly.fit_transform(df[numerical_cols])
    poly_feature_names = poly.get_feature_names_out(numerical_cols)
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)
    df = pd.concat([df, poly_df.drop(columns=numerical_cols)], axis=1)
    df.fillna(df.max(),inplace=True)

    x=df.drop(['species','species_color'],axis=1)
    y=df['species']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
    y_train.astype(int)
    y_test.astype(int)

    Lgbm_pipeline=Pipeline([('scaler',RobustScaler()),('model',LGBMClassifier(colsample_bytree=0.8,learning_rate= 0.01,max_depth= 3, min_child_weight= 1, n_estimators= 300, num_leaves= 31, reg_lambda= 3, subsample= 0.8))])

    def train_and_evaluate(model, x_train, x_test, y_train, y_test, model_name):
       
        best_model=model.fit(x_train, y_train)
        y_pred =model.predict(x_test)
        return best_model
    
   
    Lgbm_best_model= train_and_evaluate(Lgbm_pipeline,x_train,x_test,y_train,y_test, "LGBM")   
    return Lgbm_best_model
Lgbm_best_model=train_model() 

    
st.set_page_config(page_title="Iris Flower Detection", layout="wide")
st.title("🌷 Iris Flower Detection")
st.markdown("Enter the readings to check the predicted Price  of the House",width='stretch')
col1= st.columns(1)
with col1[0]:
   sepal_length = st.number_input('Sepal length',min_value=0.0,max_value=10.0,value=5.0,step=0.1)
   sepal_width = st.number_input('Sepal width',min_value=0.0,max_value=10.0,value=3.0,step=0.1)
   petal_length = st.number_input('Petal length',min_value=0.0,max_value=10.0,value=1.0,step=0.1)
   petal_width = st.number_input('Petal width',min_value=0.0,max_value=10.0,value=1.0,step=0.1)

Input_data=pd.DataFrame([[sepal_length,sepal_width,petal_length,petal_width]],columns=['sepal_length','sepal_width','petal_length','petal_width'])
Data1 = Input_data.copy()
Data1['sepal_ratio'] = Data1['sepal_length'] / (Data1['sepal_width'] + 1e-6)
Data1['petal_ratio'] = Data1['petal_length'] / (Data1['petal_width'] + 1e-6)
Data1['sepal_area'] = Data1['sepal_length'] * Data1['sepal_width']
Data1['petal_area'] = Data1['petal_length'] * Data1['petal_width']
numerical_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
poly1 = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
poly_features1= poly1.fit_transform(Data1[numerical_cols])
poly_feature_names1= poly1.get_feature_names_out(numerical_cols)
poly_Data1 = pd.DataFrame(poly_features1, columns=poly_feature_names1)
Data1 = pd.concat([Data1, poly_Data1.drop(columns=numerical_cols)], axis=1) 

if st.button("🔍Predict Price", type="primary", use_container_width=True):
    single=Lgbm_best_model.predict(Data1)
    st.markdown("---")
    for i in single:
        if (i==0).all():
            st.success('Flower is setosa')
            st.balloons()
        elif (i==1).all():
            st.success('Flower is versicolor')
            st.balloons()
        else:
            st.success('Flower is virginica')
            st.balloons()
    st.markdown("---")
    with st.expander("Input values used for prediction"):
        st.json(Input_data.to_dict(orient="records")[0])