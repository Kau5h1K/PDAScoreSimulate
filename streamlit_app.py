import streamlit as st
import numpy as np
import pandas as pd
import os
import re
import subprocess
import time
import random


def format_floats(x):
    return f'{x:.1f}' if isinstance(x, (float, int)) else x

def generate_sorted_random_integers():
    random_integers = random.sample(range(101), 4)
    random_integers.sort()
    return random_integers

def git_pull(subfolder):
    try:
        result = subprocess.run(["git", "pull", "origin", "main"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=subfolder)
        print("Output:", result.stdout)
        print("Error:", result.stderr)
    except subprocess.CalledProcessError as e:
        print("Error during git pull:", e)

def run_score_simulation(seed, n, pa, pp, ps):
    np.random.seed(seed)
    address_scores = np.random.choice([0, 2.4, 3], n, p=pa/100)
    phone_scores = np.random.choice([0, 2.4, 3], n, p=pp/100)
    specialty_scores = np.random.choice([0, 0.8, 1], n, p=ps/100)

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

def run_score_experiment(seed, num_simulations, num_records, address_auto, address_manual, address_good, phone_auto, phone_manual, phone_good, specialty_auto, specialty_manual, specialty_good):
    pa_amount = np.array([address_auto, address_manual, address_good])
    pp_amount = np.array([phone_auto, phone_manual, phone_good])
    ps_amount = np.array([specialty_auto, specialty_manual, specialty_good])

    pa = 100 * pa_amount / np.sum(pa_amount)
    pp = 100 * pp_amount / np.sum(pp_amount)
    ps = 100 * ps_amount / np.sum(ps_amount)

    simulations = [run_score_simulation(seed, num_records, pa, pp, ps) for _ in range(num_simulations)]

    average_scores = np.round(100 * np.mean(simulations, axis=0), 1)
    average_scores_df = pd.DataFrame([average_scores], columns=['Overall Address Score', 'Overall Phone Score', 'Overall Specialty Score', 'Overall Demographic Score'])

    return average_scores_df

# Sidebar inputs
with st.sidebar:
    st.subheader("Change Configuration")
    seed = st.number_input("Input the pseudo-random seed to reproduce the results", value=1, step=1, format="%d")
    num_simulations = st.slider("Select number of simulations: ", value=1000, min_value=1, max_value=10000, step=1)

random_integers = generate_sorted_random_integers()

st.title(":100: PDA Score Simulator")
st.write("Utility tool to assess the impact of score changes when performing anomalous data cleanup in ElevanceHealth's Provider Directory data (SPS).")

st.divider()

folder_path = "data"
filename_pattern = re.compile(r"ELIXIR_adhoc_PDAScoreAnalysis_(\d{8}).csv")
files = os.listdir(folder_path)

for file in files:
    match = filename_pattern.match(file)
    if match:
        timestamp = match.group(1)
        break
else:
    st.error("No matching file found.")
    st.stop()

master_df = pd.read_csv(f"./data/ELIXIR_adhoc_PDAScoreAnalysis_{timestamp}.csv")

markets = ['AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'FL', 'GA', 'IA', 'IN', 'KY', 'LA', 'MD', 'ME', 'MO', 'NH', 'NJ', 'NV', 'NY', 'NYWEST', 'OH', 'TN', 'TX', 'VA', 'WA', 'WI', 'WV']
market_selected = st.selectbox(label='Choose a market', options=markets)

filtered_df = master_df[master_df['MARKET'] == market_selected]

st.caption(f"Market Selected: {market_selected}")
st.caption(f"ProviderFinder Extract Timestamp: {filtered_df['TIMESTAMP'].iloc[0]}")

with st.sidebar:
    st.caption(f"Last Refresh Timestamp: {timestamp}")

tab1, tab2 = st.tabs(["Score Prediction", "Manual Filter Impact"])

with tab1:
    if filtered_df.empty:
        st.error(f"No matching value found for {market_selected} in data")
        st.stop()

    data = {
        'NUM_RECORDS': [filtered_df['NUM_RECORDS'].iloc[0]],
        'ADDRESS_FINAL_AUTO': [filtered_df['ADDRESS_FINAL_AUTO'].iloc[0]],
        'ADDRESS_FINAL_MANUAL': [filtered_df['ADDRESS_FINAL_MANUAL'].iloc[0]],
        'ADDRESS_FINAL_GOOD': [filtered_df['ADDRESS_FINAL_GOOD'].iloc[0]],
        'PHONE_FINAL_AUTO': [filtered_df['PHONE_FINAL_AUTO'].iloc[0]],
        'PHONE_FINAL_MANUAL': [filtered_df['PHONE_FINAL_MANUAL'].iloc[0]],
        'PHONE_FINAL_GOOD': [filtered_df['PHONE_FINAL_GOOD'].iloc[0]],
        'SPCLTY_AUTO': [filtered_df['SPCLTY_AUTO'].iloc[0]],
        'SPCLTY_MANUAL': [filtered_df['SPCLTY_MANUAL'].iloc[0]],
        'SPCLTY_GOOD': [filtered_df['SPCLTY_GOOD'].iloc[0]]
    }

    result_df = pd.DataFrame(data).transpose()
    result_df.reset_index(inplace=True)
    result_df.columns = ['Statistic', 'Volume']

    st.subheader("Data Quality Recommendations Breakdown:")
    st.info("Edit the values to tune the predictions")

    edited_df = st.data_editor(result_df, key='reco', hide_index=True)
    st.session_state['reco_bd'] = result_df
        
    if st.button('Apply Changes'):
        st.divider()
        st.subheader("Estimated Demographic Scores:")
        bar = st.progress(random_integers[0])
        time.sleep(1)
        st.session_state['reco_bd'] = edited_df

        edited_df = st.session_state['reco_bd']
        num_records = int(edited_df.loc[edited_df['Statistic'] == 'NUM_RECORDS', 'Volume'].values[0])
        address_auto = float(edited_df.loc[edited_df['Statistic'] == 'ADDRESS_FINAL_AUTO', 'Volume'].values[0])
        address_manual = float(edited_df.loc[edited_df['Statistic'] == 'ADDRESS_FINAL_MANUAL', 'Volume'].values[0])
        address_good = float(edited_df.loc[edited_df['Statistic'] == 'ADDRESS_FINAL_GOOD', 'Volume'].values[0])
        phone_auto = float(edited_df.loc[edited_df['Statistic'] == 'PHONE_FINAL_AUTO', 'Volume'].values[0])
        phone_manual = float(edited_df.loc[edited_df['Statistic'] == 'PHONE_FINAL_MANUAL', 'Volume'].values[0])
        phone_good = float(edited_df.loc[edited_df['Statistic'] == 'PHONE_FINAL_GOOD', 'Volume'].values[0])
        specialty_auto = float(edited_df.loc[edited_df['Statistic'] == 'SPCLTY_AUTO', 'Volume'].values[0])
        specialty_manual = float(edited_df.loc[edited_df['Statistic'] == 'SPCLTY_MANUAL', 'Volume'].values[0])
        specialty_good = float(edited_df.loc[edited_df['Statistic'] == 'SPCLTY_GOOD', 'Volume'].values[0])

        bar.progress(random_integers[1])
        time.sleep(1)

        average_scores_df = run_score_experiment(seed, num_simulations, num_records, address_auto, address_manual, address_good, phone_auto, phone_manual, phone_good, specialty_auto, specialty_manual, specialty_good)
        st.session_state['average_scores_df'] = average_scores_df

        bar.progress(random_integers[2])
        time.sleep(1)


        if 'average_scores_df' in st.session_state:
            data = st.session_state['average_scores_df'].round(1).to_dict(orient='records')
            bar.progress(random_integers[3])
            time.sleep(1)
            bar.progress(100)
            df_formatted = st.session_state['average_scores_df'].map(format_floats)
            st.dataframe(df_formatted, hide_index=True)
            st.warning('Tune the above attribute-wise scores to sync with Elixir-P UI for more accurate predictions')
            st.toast('Process Finished! Please review the results.')
