import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd

# Set page configuration
st.set_page_config(page_title="Hotel Booking",layout="wide",page_icon="CancelX")
    
#loading of random forest model
with open('hotel_cancellation_rf.pkl', 'rb') as file:
    rf_model = pickle.load(file)

with open('x_train_rf.pkl', 'rb') as file:
    rf_x_train = pickle.load(file)

#loading of decision tree model
with open('hotel_cancellation_dt.pkl', 'rb') as file:
    dt_model = pickle.load(file)

with open('x_train_dt.pkl', 'rb') as file:
    dt_x_train = pickle.load(file)


# sidebar for navigation
with st.sidebar:
    selected = option_menu('Hotel Cancellation Prediction System using ML',
                           
                           ['Random Forest',
                            'Decision Tree'],
                           menu_icon='building-check',
                           icons=['lightning-charge-fill', 'rocket-takeoff'],
                           default_index=0)

# Random Forest Page --------------------------------------------------------------------------------
if selected == 'Random Forest':

    # page title
    st.title('Hotel Cancellation Prediction using Random Forest')

    # Creating three columns
    col1, col2, col3 = st.columns(3)

    # Column 1 inputs
    with col1:
        hotel = st.selectbox("Select Hotel", ["Resort Hotel", "City Hotel"])
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        
        arrival_date_month = st.selectbox("Arrival Date Month", 
            ["January", "February", "March", "April", "May", "June", "July","August", "September", "October", "November", "December"])
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        
        stays_in_weekend_nights = st.number_input("Stays in Weekend Nights", min_value=0, max_value=19, value=0)
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        
        adults = st.number_input("Number of Adults", min_value=0, max_value=55, value=2)
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        
        meal = st.selectbox("Meal Type", ["BB", "FB", "HB", "SC", "Undefined"])
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        
        market_segment = st.selectbox("Market Segment", 
            ["Direct", "Corporate", "Online TA", "Offline TA/TO", "Complementary", "Groups", "Undefined", "Aviation"])
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        
        previous_cancellations = st.number_input("Previous Cancellations", min_value=0, max_value=26, value=0)
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        
        booking_changes = st.number_input("Booking Changes", min_value=0, max_value=21, value=0)
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing

        adr = st.number_input("ADR (Average Daily Rate)", min_value=0, max_value=5400, value=75)
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing

    # Column 2 inputs
    with col2:
        lead_time = st.number_input("Lead Time (days)", min_value=0, value=10)
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        
        arrival_date_year = st.selectbox("Arrival Date Year", [2015, 2016, 2017])
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        
        arrival_date_week_number = st.number_input("Arrival Date Week Number", min_value=1, max_value=53, value=1)
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        
        stays_in_week_nights = st.number_input("Stays in Week Nights", min_value=0, max_value=50, value=5)
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        
        country = st.text_input("Country Code", value="PRT")
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        
        distribution_channel = st.selectbox("Distribution Channel", 
                                            ["Direct", "Corporate", "TA/TO", "Undefined", "GDS"])
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        
        previous_bookings_not_canceled = st.number_input("Previous Bookings Not Canceled", min_value=0, max_value=72, value=0)
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing

        days_in_waiting_list = st.number_input("Days in Waiting List", min_value=0, max_value=391, value=0)
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing

        reservation_status_date = st.date_input("Reservation Status Date", value=pd.to_datetime("2015-07-01"))
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing


    # Column 3 inputs
    with col3:
        arrival_date_day_of_month = st.number_input("Arrival Date Day of Month", min_value=1, max_value=31, value=1)
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        
        babies = st.number_input("Number of Babies", min_value=0, max_value=10, value=0)
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        
        is_repeated_guest = st.selectbox("Is Repeated Guest", [0, 1])
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        
        reserved_room_type = st.selectbox("Reserved Room Type", ["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"])
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        
        assigned_room_type = st.selectbox("Assigned Room Type", ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "P"])
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        
        deposit_type = st.selectbox("Deposit Type", ["No Deposit", "Non Refund", "Refundable"])
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing

        customer_type = st.selectbox("Customer Type", ["Transient", "Contract", "Transient-Party", "Group"])
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        
        required_car_parking_spaces = st.number_input("Required Car Parking Spaces", min_value=0, max_value=8, value=0)
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing

        total_of_special_requests = st.number_input("Total of Special Requests", min_value=0, max_value=5, value=0)
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        # Prepare input data for prediction

    diagnosis=''
    if st.button('prediction Result'):
        input_data = pd.DataFrame({
            'hotel': [hotel],
            'lead_time': [lead_time],
            'arrival_date_year': [arrival_date_year],
            'arrival_date_month': [arrival_date_month],
            'arrival_date_week_number': [arrival_date_week_number],
            'arrival_date_day_of_month': [arrival_date_day_of_month],
            'stays_in_weekend_nights': [stays_in_weekend_nights],
            'stays_in_week_nights': [stays_in_week_nights],
            'adults': [adults],
            'children': [children],
            'babies': [babies],
            'meal': [meal],
            'country': [country],
            'market_segment': [market_segment],
            'distribution_channel': [distribution_channel],
            'is_repeated_guest': [is_repeated_guest],
            'previous_cancellations': [previous_cancellations],
            'previous_bookings_not_canceled': [previous_bookings_not_canceled],
            'reserved_room_type': [reserved_room_type],
            'assigned_room_type': [assigned_room_type],
            'booking_changes': [booking_changes],
            'deposit_type': [deposit_type],
            'days_in_waiting_list': [days_in_waiting_list],
            'customer_type': [customer_type],
            'adr': [adr],
            'required_car_parking_spaces': [required_car_parking_spaces],
            'total_of_special_requests': [total_of_special_requests],
            'reservation_status_date': [reservation_status_date]
        })       
        
        # One-hot encode categorical features
        input_data = pd.get_dummies(input_data, dtype=float, drop_first=True)

        # Align input data with training data columns
        missing_cols = set(rf_x_train.columns) - set(input_data.columns)
        for col in missing_cols:
            input_data[col] = 0
        input_data = input_data[rf_x_train.columns]

        # Make prediction using Random Forest model
        prediction = rf_model.predict(input_data)
        if prediction[0] == 0:
            diagnosis="The booking is predicted NOT to be canceled."
        else:
            diagnosis="The booking is predicted to be canceled."

   
    st.success(diagnosis)

#Decision Tree Model --------------------------------------------------------------------------------
if selected == 'Decision Tree':

    # page title
    st.title('Hotel Cancellation Prediction using Decision Tree')

    # Creating three columns
    col1, col2, col3 = st.columns(3)

    # Column 1 inputs
    with col1:
        hotel = st.selectbox("Select Hotel", ["Resort Hotel", "City Hotel"])
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        
        arrival_date_month = st.selectbox("Arrival Date Month", 
            ["January", "February", "March", "April", "May", "June", "July","August", "September", "October", "November", "December"])
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        
        stays_in_weekend_nights = st.number_input("Stays in Weekend Nights", min_value=0, max_value=19, value=0)
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        
        adults = st.number_input("Number of Adults", min_value=0, max_value=55, value=2)
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        
        meal = st.selectbox("Meal Type", ["BB", "FB", "HB", "SC", "Undefined"])
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        
        market_segment = st.selectbox("Market Segment", 
            ["Direct", "Corporate", "Online TA", "Offline TA/TO", "Complementary", "Groups", "Undefined", "Aviation"])
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        
        previous_cancellations = st.number_input("Previous Cancellations", min_value=0, max_value=26, value=0)
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        
        booking_changes = st.number_input("Booking Changes", min_value=0, max_value=21, value=0)
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing

        adr = st.number_input("ADR (Average Daily Rate)", min_value=0, max_value=5400, value=75)
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing

    # Column 2 inputs
    with col2:
        lead_time = st.number_input("Lead Time (days)", min_value=0, value=10)
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        
        arrival_date_year = st.selectbox("Arrival Date Year", [2015, 2016, 2017])
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        
        arrival_date_week_number = st.number_input("Arrival Date Week Number", min_value=1, max_value=53, value=1)
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        
        stays_in_week_nights = st.number_input("Stays in Week Nights", min_value=0, max_value=50, value=5)
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        
        country = st.text_input("Country Code", value="PRT")
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        
        distribution_channel = st.selectbox("Distribution Channel", 
                                            ["Direct", "Corporate", "TA/TO", "Undefined", "GDS"])
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        
        previous_bookings_not_canceled = st.number_input("Previous Bookings Not Canceled", min_value=0, max_value=72, value=0)
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing

        days_in_waiting_list = st.number_input("Days in Waiting List", min_value=0, max_value=391, value=0)
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing

        reservation_status_date = st.date_input("Reservation Status Date", value=pd.to_datetime("2015-07-01"))
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing


    # Column 3 inputs
    with col3:
        arrival_date_day_of_month = st.number_input("Arrival Date Day of Month", min_value=1, max_value=31, value=1)
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        
        babies = st.number_input("Number of Babies", min_value=0, max_value=10, value=0)
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        
        is_repeated_guest = st.selectbox("Is Repeated Guest", [0, 1])
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        
        reserved_room_type = st.selectbox("Reserved Room Type", ["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"])
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        
        assigned_room_type = st.selectbox("Assigned Room Type", ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "P"])
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        
        deposit_type = st.selectbox("Deposit Type", ["No Deposit", "Non Refund", "Refundable"])
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing

        customer_type = st.selectbox("Customer Type", ["Transient", "Contract", "Transient-Party", "Group"])
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        
        required_car_parking_spaces = st.number_input("Required Car Parking Spaces", min_value=0, max_value=8, value=0)
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing

        total_of_special_requests = st.number_input("Total of Special Requests", min_value=0, max_value=5, value=0)
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
        # Prepare input data for prediction

    diagnosis=''
    if st.button('prediction Result'):
        input_data = pd.DataFrame({
            'hotel': [hotel],
            'lead_time': [lead_time],
            'arrival_date_year': [arrival_date_year],
            'arrival_date_month': [arrival_date_month],
            'arrival_date_week_number': [arrival_date_week_number],
            'arrival_date_day_of_month': [arrival_date_day_of_month],
            'stays_in_weekend_nights': [stays_in_weekend_nights],
            'stays_in_week_nights': [stays_in_week_nights],
            'adults': [adults],
            'children': [children],
            'babies': [babies],
            'meal': [meal],
            'country': [country],
            'market_segment': [market_segment],
            'distribution_channel': [distribution_channel],
            'is_repeated_guest': [is_repeated_guest],
            'previous_cancellations': [previous_cancellations],
            'previous_bookings_not_canceled': [previous_bookings_not_canceled],
            'reserved_room_type': [reserved_room_type],
            'assigned_room_type': [assigned_room_type],
            'booking_changes': [booking_changes],
            'deposit_type': [deposit_type],
            'days_in_waiting_list': [days_in_waiting_list],
            'customer_type': [customer_type],
            'adr': [adr],
            'required_car_parking_spaces': [required_car_parking_spaces],
            'total_of_special_requests': [total_of_special_requests],
            'reservation_status_date': [reservation_status_date]
        })       
        
        # One-hot encode categorical features
        input_data = pd.get_dummies(input_data, dtype=float, drop_first=True)

        # Align input data with training data columns
        missing_cols = set(dt_x_train.columns) - set(input_data.columns)
        for col in missing_cols:
            input_data[col] = 0
        input_data = input_data[dt_x_train.columns]

        # Make prediction using Random Forest model
        prediction = dt_model.predict(input_data)
        if prediction[0] == 0:
            diagnosis="NOT to be canceled."
        else:
            diagnosis="Predicted to be Canceled."

   
    st.success(diagnosis)
