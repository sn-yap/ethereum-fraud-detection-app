import streamlit as st
import joblib
import pandas as pd
import numpy as np
from io import StringIO
import csv


# Define the base path
base_path = "./"

# Load the preprocessor pipeline and the models
preprocessor = joblib.load(base_path + 'preprocessor.joblib')
svc_model = joblib.load(base_path + 'Support Vector Classifier_original.joblib')
gb_model = joblib.load(base_path + 'Gradient Boosting Classifier_original.joblib')
rf_model = joblib.load(base_path + 'Random Forest Classifier_original.joblib')


st.title(
    "Machine Learning Defends Against Ethereum Fraud")


# Define the tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Home", "Data Upload", "Data Preprocessing", "Fraud Prediction", "Insights and Actions"])


# Content for Tab 1 - Home
with tab1:
    st.header("Home")
    st.write("""
    Use this application to predict and detect potential Ethereum fraud activities, safeguarding the integrity of the Ethereum network.

    Welcome to our Ethereum Fraud Detection Application, designed to leverage machine learning to identify suspicious transactions and potential fraud attempts on the Ethereum blockchain.
    Fraudulent activities on the Ethereum network can undermine trust and reliability, impacting users and ecosystem stakeholders. 
    Our application utilizes advanced analytics and anomaly detection to proactively defend against fraudulent behavior.
    
    #### Steps to Use the Web Application

    ##### Data Upload
    - Navigate to the data upload section.
    - Upload your Ethereum transaction data file.
    - Ensure that data contains all required fields for accurate prediction.

    ##### Data Preprocessing
    - The application will preprocess the uploaded data, handling missing values and normalizing the data as needed.
    - Review the preprocessed data summary to ensure accuracy.

    ##### Fraud Prediction
    - Initiate the fraud prediction process.
    - The application will apply machine learning algorithms to detect anomalies and flag potentially fraudulent transactions.
    - Review the prediction results, which will include labels indicating suspicious transactions.

    ##### Insights and Actions
    - Access detailed reports and visualizations to understand the characteristics of fraudulent transactions.
    - Use the insights to develop strategies for enhancing security measures and preventing fraudulent activities.
    - Export the results for further analysis or integration with existing fraud detection systems.

    #### Disclaimer
    The predictions and insights provided by our application are based on the data supplied by users and the performance of our machine learning models. 
    While we strive to deliver accurate predictions, we cannot guarantee absolute accuracy. 
    The application is intended to assist with fraud detection efforts and should be used in conjunction with other security measures. 
    Users are advised to exercise caution and employ multiple layers of protection when safeguarding Ethereum transactions.

    Thank you for choosing our Ethereum Fraud Detection Web Application to strengthen security and combat fraudulent activities on the Ethereum network. 
    Together, we can uphold the integrity of decentralized finance and blockchain technology.
    """)


# Content for Tab 2 - Data Upload
with tab2:
    st.header("Data Upload")
    st.write("Use this section to either manually input data or upload your Ethereum transaction data.")

    # Choice for the user to select input method
    input_method = st.radio("Select Input Method:", ("Manual Input", "Upload CSV"))

    # Sample template data
    template_data = {
        "time_diff_between_first_and_last_(mins)": [],
        "min_value_received": [],
        "min_value_sent_to_contract": [],
        "max_val_sent_to_contract": [],
        "total_ether_sent": [],
        "total_ether_received": [],
        "total_ether_balance": [],
        "erc20_uniq_sent_addr.1": [],
        "erc20_uniq_rec_contract_addr": []
    }

    relevant_columns = list(template_data.keys())

    def generate_template_csv():
        output = StringIO()
        csv_writer = csv.DictWriter(output, fieldnames=template_data.keys())
        csv_writer.writeheader()
        return output.getvalue()


    st.download_button(
        label="Download Template",
        data=generate_template_csv(),
        file_name="data_template.csv",
        mime="text/csv"
    )

    if input_method == "Manual Input":
        st.write("Please fill out the following fields to manually input data:")

        time_diff_between_first_and_last_mins = st.number_input("Time Diff Between First And Last (Mins)", min_value=0.0, step=0.01)
        min_value_received = st.number_input("Min Value Received", min_value=0.0, step=0.01)

        col1, col2 = st.columns(2)
        with col1:
            min_value_sent_to_contract = st.number_input("Min Value Sent To Contract", min_value=0.0, step=0.01)
        with col2:
            max_val_sent_to_contract = st.number_input("Max Value Sent To Contract", min_value=0.0, step=0.01)

        col3, col4, col5 = st.columns(3)
        with col3:
            total_ether_sent = st.number_input("Total Ether Sent", min_value=0.0, step=0.01)
        with col4:
            total_ether_received = st.number_input("Total Ether Received", min_value=0.0, step=0.01)
        with col5:
            total_ether_balance = st.number_input("Total Ether Balance", min_value=0.0, step=0.01)

        col6, col7 = st.columns(2)
        with col6:
            erc20_uniq_sent_addr = st.number_input("ERC20 Uniq Sent Addr 1", min_value=0.0, step=0.01)
        with col7:
            erc20_uniq_rec_contract_addr = st.number_input("ERC20 Uniq Rec Contract Addr", min_value=0.0, step=0.01)

        if st.button("Submit"):
            # Create a DataFrame from the input
            data = {
                "time_diff_between_first_and_last_(mins)": [time_diff_between_first_and_last_mins],
                "min_value_received": [min_value_received],
                "min_value_sent_to_contract": [min_value_sent_to_contract],
                "max_val_sent_to_contract": [max_val_sent_to_contract],
                "total_ether_sent": [total_ether_sent],
                "total_ether_received": [total_ether_received],
                "total_ether_balance": [total_ether_balance],
                "erc20_uniq_sent_addr.1": [erc20_uniq_sent_addr],
                "erc20_uniq_rec_contract_addr": [erc20_uniq_rec_contract_addr]
            }
            df = pd.DataFrame(data)


            # Store the data in the session state
            st.session_state['uploaded_data'] = df
            st.write("Data submitted successfully!")
            st.write(df)

    else:
        uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, index_col=0)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)

            st.write(df)

            # Convert column names to snake case
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

            st.write(df.columns)

            # Filter the columns to keep only the required ones
            df = df[relevant_columns]

            st.session_state['uploaded_data'] = df
            st.write("File uploaded successfully!")
            st.write(df)


# Content for Tab 3 - Data Preprocessing
with tab3:
    st.header("Data Preprocessing")
    st.write("Preprocess the uploaded data.")

    if 'uploaded_data' in st.session_state:
        df = st.session_state['uploaded_data']

        # Ensure columns are in the correct order
        numerical_cols = ['time_diff_between_first_and_last_(mins)', 'min_value_received',
                          'min_value_sent_to_contract', 'max_val_sent_to_contract',
                          'total_ether_sent', 'total_ether_received', 'total_ether_balance',
                          'erc20_uniq_sent_addr.1', 'erc20_uniq_rec_contract_addr']

        # Separate numerical data
        df_num = df[numerical_cols]

        # Display the uploaded data
        st.write("**Uploaded Data**")
        st.write(df)

        # Preprocess the data using the preprocessor pipeline and display the result
        st.write("**Preprocessed Data**")
        preprocessed_df = preprocessor.transform(df_num)
        st.write(preprocessed_df)
    else:
        st.write("Please upload your data in the 'Data Upload' tab.")


# Content for Tab 4 - Fraud Prediction
with tab4:
    st.header("Fraud Prediction")
    st.write("Upload your data to predict fraudulent transactions.")

    if 'uploaded_data' in st.session_state:
        df = st.session_state['uploaded_data']

        # Make predictions using all three models
        svc_predictions = svc_model.predict(df)
        gb_predictions = gb_model.predict(df)
        rf_predictions = rf_model.predict(df)

        if (len(df) == 1):
            # Create a row layout to display predictions side by side
            col1, col2, col3 = st.columns(3)

            # Display SVC prediction
            with col1:
                st.write("**Support Vector Classifier Prediction:**")
                if svc_predictions == 0:
                    st.success("Transaction appears legitimate.")
                else:
                    st.error("Alert: High fraud probability!")

            # Display GBC prediction
            with col2:
                st.write("**Gradient Boosting Classifier Prediction:**")
                if gb_predictions == 0:
                    st.success("Transaction appears legitimate.")
                else:
                    st.error("Alert: High fraud probability!")

            # Display RFC prediction
            with col3:
                st.write("**Random Forest Classifier Prediction:**")
                if rf_predictions == 0:
                    st.success("Transaction appears legitimate.")
                else:
                    st.error("Alert: High fraud probability!")

            # Combine predictions and determine the majority vote
            final_predictions = np.argmax(np.array([svc_predictions, gb_predictions, rf_predictions]), axis=0)

            # Display the final prediction outcome for each row
            for i, pred in enumerate(final_predictions):
                if pred == 0:
                    st.success("The majority vote indicates that this transaction appears legitimate.")
                else:
                    st.error("Alert: The majority vote indicates a high probability of fraud for this transaction.")

        else:
            # Append predictions to the DataFrame
            df['svc_predictions'] = svc_predictions
            df['gb_predictions'] = gb_predictions
            df['rf_predictions'] = rf_predictions

            # Determine the majority vote for each transaction
            final_predictions = [np.argmax(np.bincount([svc_pred, gb_pred, rf_pred]))
                                 for svc_pred, gb_pred, rf_pred in zip(svc_predictions, gb_predictions, rf_predictions)]
            df['final_predictions'] = final_predictions

            # Display the DataFrame with predictions
            st.write(df)

            # Provide a download link for the DataFrame with predictions
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Export",
                data=csv,
                file_name='predictions.csv',
                mime='text/csv',
            )


    else:
        st.write("Please upload your data in the 'Data Upload' tab.")


# Content for Tab 5 - Insights and Actions
with tab5:
    st.header("Insights and Actions")
    st.write("Generate and view detailed reports based on your data.")

    if 'uploaded_data' in st.session_state:
        df = st.session_state['uploaded_data']
        # Example report (you can replace this with your actual report logic)
        st.write("**Summary Statistics**")
        st.write(df.describe())
    else:
        st.write("Please upload your data in the 'Data Upload' tab.")
