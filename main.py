import pandas as pd
import numpy as np
from io import StringIO
import csv
import matplotlib.pyplot as plt
import joblib
import streamlit as st
import streamlit.components.v1 as components


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
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Home", "Data Upload", "Data Preprocessing", "Fraud Prediction", "Insights and Actions", "Model Performance", "Dashboard and Report"])


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

    ##### Model Performance
    - Evaluate the performance of machine learning models in detecting fraudulent transactions.
    - Explore metrics such as accuracy, precision, recall, F1-score, and ROC AUC to assess the effectiveness of each model.
    - Use visualizations to compare the performance of different models and identify the most suitable approach for fraud detection.

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
        "Index": [],
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

    relevant_columns = list(template_data.keys())[1:]  # Exclude the "Index" column

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
            # st.write(df)

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

            # st.write(df.columns)

            # Filter the columns to keep only the required ones
            df = df[relevant_columns]

            st.session_state['uploaded_data'] = df
            st.write("File uploaded successfully!")
            # st.write(df)


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
            # Counter for the number of positive predictions
            positive_predictions_counter = 0

            # Create a row layout to display predictions side by side
            col1, col2, col3 = st.columns(3)

            # Display SVC prediction
            with col1:
                st.write("**Support Vector Classifier Prediction:**")
                if svc_predictions == 0:
                    st.success("Transaction appears legitimate.")
                else:
                    st.error("Alert: High fraud probability!")
                    positive_predictions_counter += 1

            # Display GBC prediction
            with col2:
                st.write("**Gradient Boosting Classifier Prediction:**")
                if gb_predictions == 0:
                    st.success("Transaction appears legitimate.")
                else:
                    st.error("Alert: High fraud probability!")
                    positive_predictions_counter += 1

            # Display RFC prediction
            with col3:
                st.write("**Random Forest Classifier Prediction:**")
                if rf_predictions == 0:
                    st.success("Transaction appears legitimate.")
                else:
                    st.error("Alert: High fraud probability!")
                    positive_predictions_counter += 1

            # Determine the final prediction based on the majority vote
            if positive_predictions_counter > 1:
                st.error("Alert: The majority vote indicates a high probability of fraud for this transaction.")
            else:
                st.success("The majority vote indicates that this transaction appears legitimate.")

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

        # Distribution of Time Differences
        with st.expander("Distribution of Time Differences"):
            fig, ax = plt.subplots()
            ax.hist(df['time_diff_between_first_and_last_(mins)'], bins=50, edgecolor='black')
            ax.set_title('Distribution of Time Differences (mins)')
            ax.set_xlabel('Time Difference (mins)')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)

        # Min and Max Value Sent/Received
        with st.expander("Min and Max Value Sent/Received"):
            fig, ax = plt.subplots(1, 3, figsize=(18, 6))

            ax[0].hist(df['min_value_received'], bins=50, edgecolor='black')
            ax[0].set_title('Min Value Received')
            ax[0].set_xlabel('Min Value Received')
            ax[0].set_ylabel('Frequency')

            ax[1].hist(df['min_value_sent_to_contract'], bins=50, edgecolor='black')
            ax[1].set_title('Min Value Sent to Contract')
            ax[1].set_xlabel('Min Value Sent to Contract')
            ax[1].set_ylabel('Frequency')

            ax[2].hist(df['max_val_sent_to_contract'], bins=50, edgecolor='black')
            ax[2].set_title('Max Value Sent to Contract')
            ax[2].set_xlabel('Max Value Sent to Contract')
            ax[2].set_ylabel('Frequency')

            st.pyplot(fig)

        # Total Ether Sent vs. Received
        with st.expander("Total Ether Sent vs. Received"):
            fig, ax = plt.subplots()
            ax.scatter(df['total_ether_sent'], df['total_ether_received'], alpha=0.5)
            ax.set_title('Total Ether Sent vs. Received')
            ax.set_xlabel('Total Ether Sent')
            ax.set_ylabel('Total Ether Received')
            st.pyplot(fig)

        # Ether Balance Analysis
        with st.expander("Ether Balance Analysis"):
            fig, ax = plt.subplots()
            ax.hist(df['total_ether_balance'], bins=50, edgecolor='black')
            ax.set_title('Ether Balance Distribution')
            ax.set_xlabel('Total Ether Balance')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)

        # Unique ERC20 Sent and Received Addresses
        with st.expander("Unique ERC20 Sent and Received Addresses"):
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))

            ax[0].hist(df['erc20_uniq_sent_addr.1'], bins=50, edgecolor='black')
            ax[0].set_title('Unique ERC20 Sent Addresses')
            ax[0].set_xlabel('Unique ERC20 Sent Addresses')
            ax[0].set_ylabel('Frequency')

            ax[1].hist(df['erc20_uniq_rec_contract_addr'], bins=50, edgecolor='black')
            ax[1].set_title('Unique ERC20 Received Contract Addresses')
            ax[1].set_xlabel('Unique ERC20 Received Contract Addresses')
            ax[1].set_ylabel('Frequency')

            st.pyplot(fig)

        # Correlation Heatmap
        with st.expander("Correlation Heatmap"):
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = df.corr()
            cax = ax.matshow(corr, cmap='coolwarm')
            fig.colorbar(cax)
            ax.set_xticks(np.arange(len(corr.columns)))
            ax.set_yticks(np.arange(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=90)
            ax.set_yticklabels(corr.columns)
            ax.set_title('Correlation Heatmap')
            st.pyplot(fig)

    else:
        st.write("Please upload your data in the 'Data Upload' tab.")


# Content for Tab 6 - Model Performance
with tab6:
    st.header("Model Performance")
    st.write("Visualizing model performance.")

    # Load the results DataFrame from the CSV file
    results_df = pd.read_csv('results_original.csv')
    efficiency_df = pd.read_csv('Computational Efficiency.csv')


    st.write(results_df)
    st.write(efficiency_df)


    # Abbreviations for model names
    abbreviation_mapping = {
        "Support Vector Classifier": "SVC",
        "Random Forest Classifier": "RF",
        "Gradient Boosting Classifier": "GB"
    }

    # Replace model names with abbreviations
    results_df['Model'] = results_df['Model'].map(abbreviation_mapping)
    efficiency_df['Model'] = efficiency_df['Model'].map(abbreviation_mapping)


    # Train and Test Accuracy plot
    with st.expander("Train and Test Accuracy"):
        fig, ax = plt.subplots(figsize=(10, 6))
        results_df[['Model', 'Train Accuracy', 'Test Accuracy']].set_index('Model').plot(kind='bar', ax=ax)
        ax.set_ylabel('Accuracy')
        ax.set_title('Train and Test Accuracy')
        ax.legend(['Train Accuracy', 'Test Accuracy'])
        ax.set_xticklabels(results_df['Model'], rotation=0)  # Adjust the rotation angle as needed
        st.pyplot(fig)


    # Precision, Recall, and F1-Score plot
    with st.expander("Precision, Recall, and F1-Score"):
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.15
        metrics = ['Precision_0', 'Recall_0', 'F1_0', 'Precision_1', 'Recall_1', 'F1_1']
        for i, metric in enumerate(metrics):
            x = np.arange(len(results_df))
            ax.bar(x + i * bar_width, results_df[metric], width=bar_width, label=metric)
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Precision, Recall, and F1-Score')
        ax.set_xticks(np.arange(len(results_df)) + bar_width * (len(metrics) - 1) / 2)
        ax.set_xticklabels(results_df['Model'])
        ax.legend()
        st.pyplot(fig)


    # Confusion Matrix Heatmaps
    with st.expander("Confusion Matrices"):
        num_models = len(results_df)
        columns = st.columns(num_models)
        for idx, row in results_df.iterrows():
            cm = np.array([[row['True Positive'], row['False Positive']],
                           [row['False Negative'], row['True Negative']]])
            fig, ax = plt.subplots()
            cax = ax.matshow(cm, cmap='Blues')
            plt.colorbar(cax)
            for (i, j), val in np.ndenumerate(cm):
                ax.text(j, i, f'{val}', ha='center', va='center', color='black')
            ax.set_xticklabels(['', 'Predicted Positive', 'Predicted Negative'])
            ax.set_yticklabels(['', 'Actual Positive', 'Actual Negative'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'Confusion Matrix\n{row["Model"]}')
            with columns[idx]:
                st.pyplot(fig)


    # ROC AUC plot
    with st.expander("ROC AUC"):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(results_df['Model'], results_df['ROC AUC'])
        ax.set_xlabel('Model')
        ax.set_ylabel('ROC AUC')
        ax.set_title('ROC AUC')
        ax.set_xticklabels(results_df['Model'], rotation=0)
        st.pyplot(fig)

    # Training Time, Testing Time, and Inference Time plot
    with st.expander("Training Time, Testing Time, and Inference Time"):
        fig, ax = plt.subplots(figsize=(10, 6))
        efficiency_df[['Model', 'Training Time (s)', 'Testing Time (s)', 'Inference Time per Sample (ms)']].set_index('Model').plot(kind='bar', ax=ax)
        ax.set_ylabel('Time')
        ax.set_title('Training Time, Testing Time, and Inference Time')
        ax.legend(['Training Time', 'Testing Time', 'Inference Time'])
        ax.set_xticklabels(efficiency_df['Model'], rotation=0)  # Adjust the rotation angle as needed
        st.pyplot(fig)

    # Plot for Memory Usage
    with st.expander("Memory Usage"):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(efficiency_df['Model'], efficiency_df['Memory Usage (bytes)'])
        ax.set_xlabel('Model')
        ax.set_ylabel('Memory Usage (bytes)')
        ax.set_title('Memory Usage')
        ax.set_xticklabels(efficiency_df['Model'], rotation=0)
        st.pyplot(fig)

    # Plot for CPU Usage
    with st.expander("CPU Usage"):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(efficiency_df['Model'], efficiency_df['CPU Usage (%)'])
        ax.set_xlabel('Model')
        ax.set_ylabel('CPU Usage (%)')
        ax.set_title('CPU Usage')
        ax.set_xticklabels(efficiency_df['Model'], rotation=0)
        st.pyplot(fig)

    # Plot for Disk I/O
    with st.expander("Disk I/O"):
        fig, ax = plt.subplots(figsize=(10, 6))
        disk_io_read = efficiency_df['Disk I/O (Read bytes/Write bytes)'].apply(lambda x: int(x.split('/')[0].replace(' bytes', '').strip()))
        disk_io_write = efficiency_df['Disk I/O (Read bytes/Write bytes)'].apply(lambda x: int(x.split('/')[1].replace(' bytes', '').strip()))
        bar_width = 0.35
        index = np.arange(len(efficiency_df))
        ax.bar(index, disk_io_read, bar_width, label='Read bytes')
        ax.bar(index + bar_width, disk_io_write, bar_width, label='Write bytes')
        ax.set_xlabel('Model')
        ax.set_ylabel('Disk I/O (bytes)')
        ax.set_title('Disk I/O')
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(efficiency_df['Model'], rotation=0)
        ax.legend()
        st.pyplot(fig)



# Content for Tab 7 - Dashboard and Report
with tab7:
    st.header("Dashboard and Report")
    st.write("Generate and view detailed reports based on your data.")

    # Embed the Power BI dashboard
    st.components.v1.iframe(src = "https://app.powerbi.com/view?r=eyJrIjoiYzc0YjIwYmItODdlMy00YWE5LTg2YTktNTE3NTczMjUwMTQxIiwidCI6ImE2M2JiMWE5LTQ4YzItNDQ4Yi04NjkzLTMzMTdiMDBjYTdmYiIsImMiOjEwfQ%3D%3D", width = 705, height = 486)
