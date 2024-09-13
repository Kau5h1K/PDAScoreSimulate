import streamlit as st
import numpy as np
import pandas as pd
import os
import re
import subprocess
import time
import random
from PIL import Image


class GitHandler:
    @staticmethod
    def git_pull(subfolder):
        try:
            result = subprocess.run(
                ["git", "pull", "origin", "main"], check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=subfolder
            )
            print("Output:", result.stdout)
            print("Error:", result.stderr)
        except subprocess.CalledProcessError as e:
            print("Error during git pull:", e)


class ScoreSimulator:
    def __init__(self, seed, num_simulations, margin_error):
        self.seed = seed
        self.num_simulations = num_simulations
        self.margin_error = margin_error

    def format_floats(self, x):
        return f'{x:.1f}' if isinstance(x, (float, int)) else x

    def generate_sorted_random_integers(self):
        random_integers = random.sample(range(101), 4)
        random_integers.sort()
        return random_integers

    def run_simulation(self, n, pa, pp, ps):
        np.random.seed(self.seed)
        address_scores = np.random.choice([0, 2.4, 3], n, p=pa / 100)
        phone_scores = np.random.choice([0, 2.4, 3], n, p=pp / 100)
        specialty_scores = np.random.choice([0, 0.8, 1], n, p=ps / 100)

        record_max_score = 3
        row_scores = record_max_score - np.maximum.reduce([
            record_max_score - address_scores,
            record_max_score - phone_scores,
            1 - specialty_scores
        ])

        overall_address_score = np.sum(address_scores) / (n * 3)
        overall_phone_score = np.sum(phone_scores) / (n * 3)
        overall_specialty_score = np.sum(specialty_scores) / n
        overall_demographic_score = np.sum(row_scores) / (n * 3)

        return overall_address_score, overall_phone_score, overall_specialty_score, overall_demographic_score

    def run_experiment(self, num_records, pa_amount, pp_amount, ps_amount):
        pa = 100 * pa_amount / np.sum(pa_amount)
        pp = 100 * pp_amount / np.sum(pp_amount)
        ps = 100 * ps_amount / np.sum(ps_amount)

        simulations = [
            self.run_simulation(num_records, pa, pp, ps)
            for _ in range(self.num_simulations)
        ]

        average_scores = np.round(100 * np.mean(simulations, axis=0), 1)
        return pd.DataFrame([average_scores], columns=['Address Score', 'Phone Score', 'Specialty Score', 'Overall Demographic Score'])


class DataProcessor:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def get_latest_file(self):
        filename_pattern = re.compile(r"ELIXIR_adhoc_PDAScorerSummary_(\d{8}).csv")
        files = os.listdir(self.folder_path)

        for file in files:
            match = filename_pattern.match(file)
            if match:
                timestamp = match.group(1)
                return file, timestamp
        raise FileNotFoundError("No matching file found.")

    def load_data(self, filename):
        return pd.read_csv(os.path.join(self.folder_path, filename))

    @staticmethod
    def get_unique_rules(df, column_name):
        rules = df[column_name].dropna()
        rules = rules[rules.apply(lambda x: isinstance(x, str))]
        rules = rules.str.split('|').explode().unique()
        return sorted([rule for rule in rules if rule.strip()])


class AppUI:
    def __init__(self, simulator, data_processor):
        self.simulator = simulator
        self.data_processor = data_processor
        self.setup_sidebar()

    def setup_sidebar(self):
        with st.sidebar:
            st.subheader("⚙️ Change Configuration")
            st.divider()
            self.seed = st.number_input("Set a seed to reproduce random outcomes:", value=1, step=1, format="%d")
            self.num_simulations = st.slider("Select number of simulations: ", value=300, min_value=1, max_value=1000, step=1)
            self.margin_error = st.number_input("Maximum allowed margin error:", min_value=0.0, value=1.0, step=0.1)
            st.divider()
            logo = Image.open('data/hilabs_logo_v2.png')
            st.image(logo, use_column_width=True)

    def display_score_prediction(self, master_df, market_selected, num_records, address_auto, address_manual, address_good, phone_auto, phone_manual, phone_good, specialty_auto, specialty_manual, specialty_good):
        average_scores_df = self.simulator.run_experiment(
            num_records, 
            np.array([address_auto, address_manual, address_good]), 
            np.array([phone_auto, phone_manual, phone_good]), 
            np.array([specialty_auto, specialty_manual, specialty_good])
        )

        st.subheader("🧮 Estimated Demographic Scores After Cleanup:")
        st.dataframe(average_scores_df)

    def display_markets(self, master_df, market_selected):
        filtered_df = master_df[master_df['MARKET'] == market_selected]
        tab1_df = filtered_df.drop(['RULE_COMBO_LENGTH', 'RULE_TO_DISABLE', 'NUM_RECORDS_AUTOMATION_GAIN'], axis=1).drop_duplicates()
        column1 = ['Market Selected', 'ProviderFinder Extract Timestamp', 'Demographic Score', 'Network Score', 'Overall Directory Score']
        column2 = [market_selected, tab1_df['TIMESTAMP'].iloc[0], tab1_df['CURRENT_DEMOGRAPHIC_SCORE'].iloc[0], tab1_df['CURRENT_NETWORK_SCORE'].iloc[0], tab1_df['CURRENT_OVERALL_DIRECTORY_SCORE'].iloc[0]]

        table_html = """
        <div style="display: flex; justify-content: center;">
        <table>
            <tr>
            <td><b>{}</b></td><td>{}</td>
            </tr>
            <tr>
            <td><b>{}</b></td><td>{}</td>
            </tr>
            <tr>
            <td><b>{}</b></td><td>{}</td>
            </tr>
            <tr>
            <td><b>{}</b></td><td>{}</td>
            </tr>
            <tr>
            <td><b>{}</b></td><td>{}</td>
            </tr>
        </table>
        </div>
        """.format(column1[0], column2[0], column1[1], column2[1], column1[2], column2[2], column1[3], column2[3], column1[4], column2[4])

        st.markdown(table_html, unsafe_allow_html=True)

    def run(self):
        st.title("🎯 Provider Directory Score Guide")
        st.write("Utility tool to assess the impact of score changes when performing anomalous data cleanup in provider directory data.")

        folder_path = "data"
        try:
            latest_file, timestamp = self.data_processor.get_latest_file()
            master_df = self.data_processor.load_data(latest_file)
        except FileNotFoundError as e:
            st.error(str(e))
            return

        with st.sidebar:
            st.caption(f"⌚ Last Refresh Timestamp: **{timestamp}**")
            st.caption("💻 Developed by **HiLabs**")

        tab1, _, _, _ = st.tabs([":star: Score Prediction", "📉 Manual Filter Impact", "🌐 Directory 360°", "❓ Scoring Methodology"])
        with tab1:
            market_selected = st.selectbox(label='Choose a market', options=master_df['MARKET'].unique())
            num_records = st.slider("Number of Records:", 1, 1000, 100)
            address_auto, address_manual, address_good = st.slider("Address Scores (Auto, Manual, Good)", 0.0, 100.0, (30.0, 30.0, 40.0))
            phone_auto, phone_manual, phone_good = st.slider("Phone Scores (Auto, Manual, Good)", 0.0, 100.0, (20.0, 30.0, 50.0))
            specialty_auto, specialty_manual, specialty_good = st.slider("Specialty Scores (Auto, Manual, Good)", 0.0, 100.0, (10.0, 40.0, 50.0))

            self.display_score_prediction(master_df, market_selected, num_records, address_auto, address_manual, address_good, phone_auto, phone_manual, phone_good, specialty_auto, specialty_manual, specialty_good)


# Instantiate the classes and run the app
simulator = ScoreSimulator(seed=1, num_simulations=300, margin_error=1.0)
data_processor = DataProcessor(folder_path="data")
app_ui = AppUI(simulator, data_processor)
app_ui.run()
