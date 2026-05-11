
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import anthropic

st.set_page_config(
    page_title="Hotel Cancellation Risk Tool",
    page_icon="🏨",
    layout="wide"
)

st.title("🏨 Hotel Booking Cancellation Risk Tool")
st.markdown("*Powered by XGBoost + Claude AI — for Revenue Managers*")

# Load model and features
@st.cache_resource
def load_model():
    model = joblib.load("xgb_model.pkl")
    features = joblib.load("feature_columns.pkl")
    return model, features

model, feature_columns = load_model()

# Load API client
@st.cache_resource
def load_client():
    return anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

client = load_client()

# Sidebar inputs
st.sidebar.header("Enter Booking Details")

hotel = st.sidebar.selectbox("Hotel Type", ["City Hotel", "Resort Hotel"])
lead_time = st.sidebar.slider("Lead Time (days before arrival)", 0, 700, 90)
adr = st.sidebar.number_input("Average Daily Rate ($)", 0.0, 1000.0, 120.0)
total_special_requests = st.sidebar.slider("Total Special Requests", 0, 5, 0)
required_parking = st.sidebar.selectbox(
    "Requires Parking?", [0, 1], 
    format_func=lambda x: "Yes" if x == 1 else "No"
)
previous_cancellations = st.sidebar.slider(
    "Previous Cancellations on Record", 0, 10, 0
)
booking_changes = st.sidebar.slider("Booking Changes Made", 0, 10, 0)
deposit_type = st.sidebar.selectbox(
    "Deposit Type", ["No Deposit", "Non Refund", "Refundable"]
)
customer_type = st.sidebar.selectbox(
    "Customer Type", ["Transient", "Transient-Party", "Contract", "Group"]
)
market_segment = st.sidebar.selectbox(
    "Market Segment", 
    ["Online TA", "Offline TA/TO", "Direct", "Corporate", "Groups"]
)

# Encode inputs to match training data
deposit_map = {"No Deposit": 1, "Non Refund": 0, "Refundable": 2}
customer_map = {"Transient": 3, "Transient-Party": 2, "Contract": 0, "Group": 1}
segment_map = {"Online TA": 4, "Offline TA/TO": 3, "Direct": 2, 
               "Corporate": 1, "Groups": 0}
hotel_map = {"City Hotel": 0, "Resort Hotel": 1}
meal_map = {"BB": 0, "FB": 1, "HB": 2, "SC": 3, "Undefined": 4}

# Build input dataframe with all expected columns
input_dict = {col: 0 for col in feature_columns}
input_dict.update({
    "hotel": hotel_map[hotel],
    "lead_time": lead_time,
    "arrival_date_year": 2024,
    "arrival_date_month": 6,
    "arrival_date_week_number": 25,
    "arrival_date_day_of_month": 15,
    "stays_in_weekend_nights": 1,
    "stays_in_week_nights": 2,
    "adults": 2,
    "children": 0,
    "babies": 0,
    "meal": 1,
    "country": 50,
    "market_segment": segment_map[market_segment],
    "distribution_channel": 1,
    "is_repeated_guest": 0,
    "previous_cancellations": previous_cancellations,
    "previous_bookings_not_canceled": 0,
    "reserved_room_type": 0,
    "assigned_room_type": 0,
    "booking_changes": booking_changes,
    "deposit_type": deposit_map[deposit_type],
    "agent": 0,
    "days_in_waiting_list": 0,
    "customer_type": customer_map[customer_type],
    "adr": adr,
    "required_car_parking_spaces": required_parking,
    "total_of_special_requests": total_special_requests
})

input_df = pd.DataFrame([input_dict])[feature_columns]

# Main layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Cancellation Risk Score")

    if st.button("Analyze Booking", type="primary"):
        risk_score = model.predict_proba(input_df)[0][1]

        if risk_score > 0.6:
            color = "red"
            label = "HIGH RISK"
        elif risk_score > 0.3:
            color = "orange"
            label = "MEDIUM RISK"
        else:
            color = "green"
            label = "LOW RISK"

        st.markdown(f"### :{color}[{label}]")
        st.metric("Cancellation Probability", f"{risk_score:.1%}")

        # Risk gauge
        st.progress(float(risk_score))

        # Claude recommendation
        st.subheader("AI Recommendation")
        with st.spinner("Getting Claude recommendation..."):
            prompt = f"""You are a hotel revenue management assistant.

Cancellation Risk Score: {risk_score:.1%}
Booking details:
- Lead time: {lead_time} days
- Average daily rate: ${adr}
- Special requests: {total_special_requests}
- Deposit type: {deposit_type}
- Customer type: {customer_type}
- Market segment: {market_segment}
- Previous cancellations: {previous_cancellations}
- Parking required: {"Yes" if required_parking else "No"}

Write a 3-part response:
1. RISK SUMMARY: One sentence explaining the risk level
2. KEY DRIVERS: 2-3 bullet points explaining the main reasons
3. RECOMMENDED ACTION: One specific actionable recommendation

Under 150 words. Be direct and practical."""

            message = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            st.markdown(message.content[0].text)

with col2:
    st.subheader("Natural Language Booking Search")
    st.markdown("Ask any question about your bookings in plain English.")

    query = st.text_input(
        "Search bookings...",
        placeholder="e.g. Show me high risk bookings with no deposit"
    )

    if query:
        with st.spinner("Searching..."):
            prompt = f"""Convert this hotel booking query to a pandas filter expression.
Query: "{query}"
DataFrame name: df, available columns: lead_time (int), adr (float), 
cancellation_risk (float 0-1), previous_cancellations (int), 
total_of_special_requests (int), deposit_type (str), customer_type (str),
hotel (str), market_segment (str).
Risk levels: high > 0.6, medium 0.3-0.6, low < 0.3.
Return ONLY the filter expression, no explanation, no markdown."""

            message = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}]
            )

            filter_expr = message.content[0].text.strip()
            st.markdown("**Interpreted filter:**")
            st.code(filter_expr)
            st.info(
                "In full deployment this query runs against your live "
                "property management system and returns matching bookings."
            )

st.markdown("---")
st.markdown(
    "*Model: XGBoost (ROC-AUC 0.92) | "
    "AI: Claude (Anthropic) | "
    "Data: Hotel Booking Demand Dataset*"
)
