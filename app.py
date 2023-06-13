import streamlit as st
# from IPython.display import display, HTML
import pandas as pd 
import numpy as np
import pickle
import base64


with open("hotel_bg.jpg", "rb") as image_file:
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

st.title("Hotel Booking Cancellation Prediction")
st.markdown("Will this customer honour the booking? ")

# step 1 load the pickled model --> rb read binary

with open("oe_market.pkl", "rb") as file:
    encode_market = pickle.load(file)

with open("oe_roomtype.pkl", "rb") as file:
    encode_roomtype = pickle.load(file)
    
with open("oe_mealplan.pkl", "rb") as file:
    encode_mealplan = pickle.load(file)
    
with open("avg_price_per_room.pkl", "rb") as file:
    transform_avgprice = pickle.load(file)

with open("no_of_previous_bookings_not_canceled.pkl", "rb") as file:
    transform_prevbooking = pickle.load(file)

with open("no_of_previous_cancellations.pkl", "rb") as file:
    transform_prevcan = pickle.load(file)
    
with open("lead_time.pkl", "rb") as file:
    transform_leadtime = pickle.load(file)

with open("no_of_week_nights.pkl", "rb") as file:
    transform_weeknights = pickle.load(file)

with open("no_of_weekend_nights.pkl", "rb") as file:
    transform_weekendnights = pickle.load(file)

with open("no_of_children.pkl", "rb") as file:
    transform_children = pickle.load(file)

with open("no_of_adults.pkl", "rb") as file:
    transform_adults = pickle.load(file)

with open("xg.pkl", "rb") as file:
    model_xg = pickle.load(file)


    
# step2 get the user input from the front end
no_of_adults= st.number_input('No of Adults',0,4,step = 1) 
no_of_children = st.slider('No of Children',0,10,1) 
no_of_weekend_nights = st.slider("No of weekend nights",0,7,1)
no_of_week_nights = st.slider('No of week nights',0,17,1)
type_of_meal_plan = st.selectbox("Select a meal plan ", ('Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'))
required_car_parking_space = st.selectbox("Parking required or not ", ("Yes", "No"))
room_type_reserved = st.selectbox("Type of room type reserved ", ('Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4',
                       'Room_Type 5', 'Room_Type 6', 'Room_Type 7'))
lead_time = st.number_input("Lead Time" , 0,443,1)
arrival_month = st.slider("Month of arrival " , 1,12,1)
arrival_date = st.slider("Date of arrival", 1,30,1)
market_segment_type = st.selectbox("Mode of Booking ", ('Online','Aviation','Offline','Corporate','Complementary'))
repeated_guest = st.selectbox("Repeat visit" , ("Yes", "No"))
no_of_previous_cancellations = st.slider("No of previous cancellations", 0,13,1)
no_of_previous_bookings_not_canceled = st.slider("No of successful visits" , 0,58,1)
avg_price_per_room = st.slider("Price per room" , 0, 540, 10)
no_of_special_requests = st.slider("Special requests if any" , 0,5,1)


# step 3 : doing encxoding / scaling / transformationn on user iput data

adults = transform_adults.transform(pd.DataFrame([no_of_adults]))
children = transform_children.transform(pd.DataFrame([no_of_children]))
weekend = transform_weekendnights.transform(pd.DataFrame([no_of_weekend_nights]))
week = transform_weeknights.transform(pd.DataFrame([no_of_week_nights]))
mealplan = encode_mealplan.transform(pd.DataFrame([type_of_meal_plan]))
parking = [1 if required_car_parking_space == "Yes" else 0]
roomtype = encode_roomtype.transform(pd.DataFrame([room_type_reserved]))
lead = transform_leadtime.transform(pd.DataFrame([lead_time]))
market = encode_market.transform(pd.DataFrame([market_segment_type]))
repeat_guest = [1 if  repeated_guest =="Yes" else 0]
prevcancelled = transform_prevcan.transform(pd.DataFrame([no_of_previous_cancellations ]))
prevnotcanc = transform_prevbooking.transform(pd.DataFrame([no_of_previous_bookings_not_canceled]))
price = transform_avgprice.transform(pd.DataFrame([avg_price_per_room]))

data = {'no_of_adults': adults[0],
        'no_of_children' : children[0], 
        'no_of_weekend_nights' : weekend[0], 
        'no_of_week_nights': week[0],
        'type_of_meal_plan' : mealplan[0],
       'required_car_parking_space': parking,
        'room_type_reserved': roomtype[0],
        'lead_time':lead[0],
       "arrival_month": arrival_month,
       "arrival_date": arrival_date,
       "market_segment_type": market[0],
       "repeated_guest": repeat_guest,
       "no_of_previous_cancellations" : prevcancelled[0],
       "no_of_previous_bookings_not_canceled" : prevnotcanc[0],
       "avg_price_per_room": price[0],
       "no_of_special_requests" : no_of_special_requests}
input_data = pd.DataFrame(data)
# st.write(input_data)



prediction = model_xg.predict(input_data)

if st.button("Prediction"):
    if prediction == 1:
        st.subheader("Booking will be honoured")
    if prediction== 0 :
        st.subheader("Booking will be cancelled")
