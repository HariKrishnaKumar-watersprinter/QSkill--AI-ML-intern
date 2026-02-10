import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from  lightgbm import LGBMRegressor
from imblearn.pipeline import Pipeline
from sklearn.metrics import r2_score

DATA_PATH = r'F:\Project\Hosue price Prediction\New folder (2)\Housing_Price_Data.csv'
@st.cache_resource
def train_model():
    df=pd.read_csv(DATA_PATH)
    binary_cols = ['Main road', 'guestroom', 'basement', 'Hot water heating', 'Airconditioning', 'Preferred area']
    for col in binary_cols:
        if df[col].dtype == 'object':
            df[col] = df[col].map({'yes': 1, 'no': 0})
    def cap_outliers_iqr(df, column, iqr_factor=1.5):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
    
        lower_bound = Q1 - iqr_factor * IQR
        upper_bound = Q3 + iqr_factor * IQR
    
    
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    
    
        return df
    important_cols = [ 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']
    for col in important_cols:
        df = cap_outliers_iqr(df, col, iqr_factor=1.5)
    df_fe = df.copy()
    def get_logprice(area):
        if area <= 3450:
            return 14.375  + (area / 3450) * 0.8125          
        elif area > 3450 & area <= 4050:
            return 15.238 + ((area-3450)/(4050-3450)) * (16.0195 - 15.7125)
        elif area > 4050 & area <= 5400:
            return 15.476+ ((area-4050)/(5400-4050)) * (16.068 - 16.0895)
        elif area > 5400 & area <= 6600:
            return 15.722 + ((area-5400)/(6600-5400)) * (16.25 - 16.068)
        elif area > 6600:
            return 16.25 + ((area-6600)/1000) * 0.1
    df_fe ['price_log'] = np.vectorize(get_logprice)(df_fe['area']) 
    df_fe['area_per_room'] = df_fe['area'] / (df_fe['bedrooms'] + df_fe['bathrooms'].clip(lower=1))
    df_fe['area_per_bedroom'] = df_fe['area'] / df_fe['bedrooms'].clip(lower=1)
    df_fe['area_per_bathroom'] = df_fe['area'] / df_fe['bathrooms'].clip(lower=1)
    df_fe['small_house'] = (df_fe['area'] < 1000).astype(int)          
    df_fe['large_house'] = (df_fe['area'] > 3000).astype(int)
    df_fe['total_rooms']      = df_fe['bedrooms'] + df_fe['bathrooms']
    df_fe['bed_bath_ratio']   = df_fe['bedrooms'] / df_fe['bathrooms'].clip(lower=1)
    df_fe['bath_per_bedroom'] = df_fe['bathrooms'] / df_fe['bedrooms'].clip(lower=1)
    amenity_cols = ['Main road', 'guestroom', 'basement', 'Hot water heating', 
                'Airconditioning', 'Preferred area']
    df_fe['amenity_score'] = df_fe[amenity_cols].sum(axis=1)
    df_fe['luxury_score'] = (
     df_fe['Airconditioning'] +
     df_fe['Preferred area']  +
     df_fe['guestroom']      +
     df_fe['basement']         +
     df_fe['Hot water heating']  +
     df_fe['Main road'])
    df_fe['stories_parking'] = df_fe['stories'] * df_fe['parking']
    df_fe['has_parking']     = (df_fe['parking'] > 0).astype(int)
    df_fe['high_parking']    = (df_fe['parking'] >= 2).astype(int)
    furnish_map = {'unfurnished':0,'semi-furnished':1,'furnished': 2}
    df_fe['furnishing_score'] = df_fe['Furnishing status'].map(furnish_map)
    df_fe['area_x_aircon']      = df_fe['area'] * df_fe['Airconditioning']
    df_fe['area_x_prefarea']    = df_fe['area'] * df_fe['Preferred area']
    df_fe['area_x_luxury']      = df_fe['area'] * df_fe['luxury_score']
    df_fe['bedrooms_x_aircon']  = df_fe['bedrooms'] * df_fe['Airconditioning']
    for col in ['area', 'parking']:
        df_fe[f'log_{col}']  = np.log1p(df_fe[col])
    df_fe['area_bin'] = pd.qcut(df_fe['area'], q=5, labels=False, duplicates='drop')
    df_fe['price_bin'] = pd.qcut(df_fe['price'], q=5, labels=False)
    x=df_fe.drop(['price', 'Furnishing status','bed_bath_ratio','area_bin'], axis=1)
    y=df_fe['price']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=None)
    def evaluate_model(model,x_train1,X_test1,y_train1,y_test1):
        model.fit(x_train1,y_train1)
        y_pred = model.predict(X_test1)
        Metrics={'r2': r2_score(y_test1, y_pred)}
        print(Metrics)
        return model
    lgb_pipeline = Pipeline(steps=[('scaler', RobustScaler()),('model', LGBMRegressor( objective='regression',metric='rmse',random_state=42, verbosity=-1,colsample_bytree=1.0, learning_rate= 0.1, max_depth= 10, min_split_gain= 0.1, n_estimators= 30, num_leaves= 31,subsample= 0.3))])
    lgbm_model= evaluate_model(lgb_pipeline,x_train,x_test,y_train,y_test)
    return lgbm_model
lgbm_model= train_model()
st.set_page_config(page_title="House Price Predictor", layout="wide")
st.title("🏚️ House Price Predictor")
st.markdown("Enter the readings to check the predicted Price  of the House",width='stretch')
col1,col2= st.columns(2)
with col1:
    Area = st.number_input("Area (in sqft)", min_value=100, max_value=30000, value=1500, step=50)
    Bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3, step=1)
    Bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2, step=1)
    Stories = st.number_input("Number of Stories", min_value=1, max_value=10, value=1, step=1)
    Mainroad= st.selectbox("Main Road", options=['Yes', 'No'])
    Guestroom= st.selectbox("Guest Room", options=['Yes', 'No'])
with col2:
    Basement= st.selectbox("Basement", options=['Yes', 'No'])
    Hotwaterheating= st.selectbox("Hot Water Heating", options=['Yes', 'No'])
    Airconditioning= st.selectbox("Air Conditioning", options=['Yes', 'No'])
    Parking= st.number_input("Parking", min_value=0, max_value=10, value=1, step=1)
    Preferredarea= st.selectbox("Preferred Area", options=['Yes', 'No'])
    Furnishingstatus= st.selectbox("Furnishing Status", options=['Unfurnished', 'Semi-furnished', 'Furnished'])

Input_data = pd.DataFrame([[Area, Bedrooms, Bathrooms, Stories, Mainroad, Guestroom, Basement, Hotwaterheating, Airconditioning, Parking, Preferredarea, Furnishingstatus]], 
                          columns=['area', 'bedrooms', 'bathrooms', 'stories', 'Main road', 'guestroom', 'basement', 'Hot water heating', 'Airconditioning', 'parking', 'Preferred area', 'Furnishing status'])
binary_cols1= ['Main road', 'guestroom', 'basement', 'Hot water heating', 'Airconditioning', 'Preferred area']
for col in binary_cols1:
    if Input_data[col].dtype == 'object':
        Input_data[col] = Input_data[col].map({'Yes': 1, 'No': 0})
Data1 = Input_data.copy()
def get_logprice(area):
    if area <= 3450:
        return 14.375  + (area / 3450) * 0.8125          
    elif area > 3450 & area <= 4050:
        return 15.238 + ((area-3450)/(4050-3450)) * (16.0195 - 15.7125)
    elif area > 4050 & area <= 5400:
        return 15.476+ ((area-4050)/(5400-4050)) * (16.068 - 16.0895)
    elif area > 5400 & area <= 6600:
        return 15.722 + ((area-5400)/(6600-5400)) * (16.25 - 16.068)
    elif area > 6600:
        return 16.25 + ((area-6600)/1000) * 0.1
Data1['price_log'] = np.vectorize(get_logprice)(Data1['area'])
Data1['area_per_room'] = Data1['area'] / (Data1['bedrooms'] + Data1['bathrooms'].clip(lower=1)) 
Data1['area_per_bedroom'] = Data1['area'] / Data1['bedrooms'].clip(lower=1)
Data1['area_per_bathroom'] = Data1['area'] / Data1['bathrooms'].clip(lower=1)
Data1['small_house'] = (Data1['area'] < 1000).astype(int)         
Data1['large_house'] = (Data1['area'] > 3000).astype(int)
Data1['total_rooms'] = Data1['bedrooms'] + Data1['bathrooms']
Data1['bed_bath_ratio'] = Data1['bedrooms'] / Data1['bathrooms'].clip(lower=1)
Data1['bath_per_bedroom'] = Data1['bathrooms'] / Data1['bedrooms'].clip(lower=1)
amenity_cols = ['Main road', 'guestroom', 'basement', 'Hot water heating', 
                'Airconditioning', 'Preferred area']
Data1['amenity_score'] = Data1[amenity_cols].sum(axis=1)

Data1['luxury_score'] = (
    Data1['Airconditioning'] +
    Data1['Preferred area']  +
    Data1['guestroom']      +
    Data1['basement']         +
    Data1['Hot water heating']  +
    Data1['Main road'] )

Data1['stories_parking'] = Data1['stories'] * Data1['parking']
Data1['has_parking']     = (Data1['parking'] > 0).astype(int)
Data1['high_parking']    = (Data1['parking'] >= 2).astype(int)

furnish_map = {'Unfurnished': 0,'Semi-furnished':1,'Furnished':2}

Data1['furnishing_score'] = Data1['Furnishing status'].map(furnish_map)
Data1['area_x_aircon']      = Data1['area'] * Data1['Airconditioning']
Data1['area_x_prefarea']    = Data1['area'] * Data1['Preferred area']
Data1['area_x_luxury']      = Data1['area'] * Data1['luxury_score']
Data1['bedrooms_x_aircon']  = Data1['bedrooms'] * Data1['Airconditioning']
for col in ['area', 'parking']:
    Data1[f'log_{col}']  = np.log1p(Data1[col])

Data1['area_bin'] = pd.qcut(Data1['area'], q=5, labels=False, duplicates='drop')


conditions_bin = [
    Data1['area'] <= 3450,
    (Data1['area'] > 3450) & (Data1['area'] <= 4050),
    (Data1['area'] > 4050) & (Data1['area'] <= 5400),
    (Data1['area'] > 5400) & (Data1['area'] <= 6600),
    Data1['area'] > 6600]

choices_bin = [0, 1, 2, 3, 4]

Data1['price_bin'] = np.select(conditions_bin, choices_bin,default=np.nan)
Data1.drop(['Furnishing status','bed_bath_ratio','area_bin'], axis=1, inplace=True)

if st.button("🔍Predict Price", type="primary", use_container_width=True):
    
    single=lgbm_model.predict(Data1)
    print(single)
    st.markdown("---")
    st.success(f"✅ House Price is: {single[0].astype(int)}")
    st.balloons()
    st.markdown("---")
    with st.expander("Input values used for prediction"):
        st.json(Input_data.to_dict(orient="records")[0])
    with st.expander("Input values used for prediction"):
        st.json(Data1.to_dict(orient="records")[0])