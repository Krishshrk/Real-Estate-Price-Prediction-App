import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import base64
@st.cache(allow_output_mutation=True)
def preprocess():
    pass

#feature engineering
data=pd.read_csv("Housing_Data.csv")
data['roomsPerHousehold']=data["totalRooms"]/data["households"]
data['populationPerHousehold']=data["population"]/data["households"]
data['bedroomsPerRoom']=data["totalBedrooms"]/data["totalRooms"]
data=data.drop(['longitude','latitude','housingMedianAge','totalRooms'],axis=1)

#data preprocessing
X=data.drop('medianHouseValue',axis=1)
y=data['medianHouseValue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
sc_x=StandardScaler()
sc_y=StandardScaler()   
X_train=sc_x.fit_transform(X_train)
y_train=np.array(y_train).reshape(y_train.shape[0],1)
y_train=sc_y.fit_transform(y_train)
   

#streamlit section
st.set_page_config(
    page_title="Real Estate Price Prediction",
    page_icon="https://i.pinimg.com/originals/08/56/22/085622af8231ab6eb78b80eb2add557a.jpg",
    layout="centered",
    initial_sidebar_state="expanded")
    

page_bg_img='''
    <style>
    .stApp{
    background-image: url('https://wallpapers.com/images/hd/cool-photos-background-1920-x-1200-6mdrogrwtemt2rhx.jpg');
    background-size:cover;
    }
    </style>
    '''
    
st.markdown(page_bg_img,unsafe_allow_html=True)

app_mode=st.sidebar.selectbox('Select Page',['Welcome','Home','Prediction'])
if app_mode=='Welcome':
    st.markdown("<h2 style='color:white;text-align:center;margin-bottom:5px'>Welcome to Real Estate Housing Price Prediction Web App</h2>",unsafe_allow_html=True)
    st.markdown("<h3 style='color:lightgreen;text-align:center'>Home Page : Information about website and dataset</h3>",unsafe_allow_html=True)
    st.markdown("<h3 style='color:lightyellow;text-align:center'>Prediction Page: Predict the real estate housing price with user input feature values</h3>",unsafe_allow_html=True)
elif app_mode=='Home':
    st.snow()
    st.markdown("<h2 style='text-align:center;margin-bottom:5px'>Real Estate Hosing Price Prediction</h2>",unsafe_allow_html=True)
    st.markdown("<h3 style='color:lightblue;text-align:center'>Want to know the land prices for a housing plot move on to Prediction Page</h3>",unsafe_allow_html=True)
    st.image('real_estate_image.jpg')
    st.markdown("<h4 style='color:lightgreen'>Used Dataset Sample:</h4>",unsafe_allow_html=True)
    st.write(data.head())
    st.markdown("<h4 style='color:lightgreen'>Dataset statistical information is :</h4>",unsafe_allow_html=True)
    st.write(data.describe())
elif app_mode=='Prediction':
    st.markdown("<h2 style='text-align:center;margin-bottom:5px'>Welcome to prediction page</h2>",unsafe_allow_html=True)
    st.image('img.jpg')
    st.markdown("<h2 style='text-align:center'>Fill all necessary information in order to get the real estate housing price estimated amount</h2>",unsafe_allow_html=True)
    pred=st.button("Predict The Housing Price")
    st.sidebar.header("Information about the housing land")
    #latitude=st.sidebar.number_input("Latitude of the Location")
    #longitude=st.sidebar.number_input("Longitude of the Location")
    #age=st.sidebar.number_input("Median Age group of the people in Housing Area",10,100)
    room=st.sidebar.number_input("Total rooms in the Housing Area")
    bedroom=st.sidebar.number_input("Total Bedroom in the Housing Area")
    population=st.sidebar.number_input("Population strength in Housing Area")
    house=st.sidebar.number_input("Total Number of Houses in Housing Area")
    income=st.sidebar.number_input("Median Income of the people in Housing Area($)")
    if room and bedroom and population and house and income:
        val=[bedroom,population,house,income,room/house,population/house,bedroom/room]
        val=list(map(lambda x:float(x),val))
        val=np.array(val)
        val=val.reshape(1,7)
        val=pd.DataFrame(val,columns=X.columns)
        val_new=sc_x.transform(val)
        if pred:
            file=open("price.gif","rb")
            contents=file.read()
            data_url=base64.b64encode(contents).decode("utf-8")
            file.close()
            loaded_model=pickle.load(open('real_estate_model_gbr.sav','rb'))
            prediction=loaded_model.predict(val_new)
            st.success("The Estimated Price for Housing Area is " + str(round(sc_y.inverse_transform([prediction])[0][0],3)) + "$")
            st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="price gif">',unsafe_allow_html=True,)
            
        
    
    
    