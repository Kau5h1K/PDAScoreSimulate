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
    average_scores_df = pd.DataFrame([average_scores], columns=['Address Score', 'Phone Score', 'Specialty Score', 'Overall Demographic Score'])

    return average_scores_df

# Sidebar inputs
with st.sidebar:
    st.subheader("⚙️ Change Configuration")
    st.divider()
    seed = st.number_input("Set a seed to reproduce random outcomes:", value=1, step=1, format="%d")
    with st.expander(":bulb: Info"):
        st.info('''
        Setting a seed ensures consistent and reproducible random outcomes each time the code runs.
        ''')
    st.divider()
    num_simulations = st.slider("Select number of simulations: ", value=300, min_value=1, max_value=1000, step=1)
    with st.expander(":bulb: Info"):
        st.info('''
        Increasing the number of simulations improves the accuracy of score predictions but also extends the runtime.
        ''')
    st.divider()
    margin_error = st.number_input("Maximum allowed margin error:", min_value=0.0, value=1.0, step=0.1)
    with st.expander(":bulb: Info"):
        st.info('''
        The margin error determines the maximum allowed difference between the **actual demographic score** and the **simulated demographic score**.
        ''')

variance_flag = 0
improvement_flag = 0
random_integers = generate_sorted_random_integers()

st.title(":100: PDA Score Impact Assessment")
st.write("Utility tool to assess the impact of score changes when performing anomalous data cleanup in ElevanceHealth's Provider Directory data (SPS).")

folder_path = "data"
filename_pattern = re.compile(r"ELIXIR_adhoc_PDAScorerSummary_(\d{8}).csv")
files = os.listdir(folder_path)

for file in files:
    match = filename_pattern.match(file)
    if match:
        timestamp = match.group(1)
        break
else:
    st.error("No matching file found.")
    st.stop()

with st.sidebar:
    st.divider()
    st.caption(f"⌚ Last Refresh Timestamp: {timestamp}")
    st.caption("💻 Developed by **HiLabs**")


master_df = pd.read_csv(f"./data/ELIXIR_adhoc_PDAScorerSummary_{timestamp}.csv")

tab1, tab2 = st.tabs([":star: Score Prediction", ":fire: Manual Filter Impact"])

with tab1:
    st.text("")
    with st.expander(":bulb: See instructions"):
        st.write('''
        **Instructions:**
        - Select a market under the market dropdown menu.
        - In the **Data Quality Recommendations Breakdown** table, click on any cell under the "Volume" column so that it's highlighted.
        - Type the updated amount to change the recommendation volume and hit "Enter".
        - Click on "Apply Changes" below the table to view the Demographic score impact.
        ---
                 ''')
        st.info('''                 
        The **Data Quality Recommendations Breakdown** table below shows the breakdown of data quality recommendations for each attribute - address, phone, specialty.
        > The number of records represents the unique NPI-Address entries within the scope of directory validation for a specific market.
        ''')
    st.text("")
    markets = ['AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'FL', 'GA', 'IA', 'IN', 'KY', 'LA', 'MD', 'ME', 'MO', 'NH', 'NJ', 'NV', 'NY', 'NYWEST', 'OH', 'TN', 'TX', 'VA', 'WA', 'WI', 'WV']
    market_selected = st.selectbox(label='Choose a market', options=markets)

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

    # Display the table
    st.markdown(table_html, unsafe_allow_html=True)
    st.divider()

    if tab1_df.empty:
        st.error(f"No matching value found for {market_selected} in data")
        st.stop()

    data = {
        'Number of Records': [tab1_df['NUM_RECORDS'].iloc[0]],
        'Address - Auto Update (Final Suggestion)': [tab1_df['ADDRESS_FINAL_AUTO'].iloc[0]],
        'Address - Manual Review (Final Suggestion)': [tab1_df['ADDRESS_FINAL_MANUAL'].iloc[0]],
        'Address - Good (Final Suggestion)': [tab1_df['ADDRESS_FINAL_GOOD'].iloc[0]],
        'Phone - Auto Update (Final Suggestion)': [tab1_df['PHONE_FINAL_AUTO'].iloc[0]],
        'Phone - Manual Review (Final Suggestion)': [tab1_df['PHONE_FINAL_MANUAL'].iloc[0]],
        'Phone - Good (Final Suggestion)': [tab1_df['PHONE_FINAL_GOOD'].iloc[0]],
        'Specialty - Auto Update': [tab1_df['SPCLTY_AUTO'].iloc[0]],
        'Specialty - Manual Review': [tab1_df['SPCLTY_MANUAL'].iloc[0]],
        'Specialty - Good': [tab1_df['SPCLTY_GOOD'].iloc[0]]
    }

    result_df = pd.DataFrame(data).transpose()
    result_df.reset_index(inplace=True)
    result_df.columns = ['Statistic', 'Volume']
    st.subheader("📊 Data Quality Recommendations Breakdown:")
    st.text("")

    col1, col2, col3 = st.columns([1, 3, 0.25])  # Adjust the width ratio to your needs

    with col2:
        edited_df = st.data_editor(result_df, key='reco', hide_index=True, column_config={
        "Statistic": st.column_config.TextColumn(disabled=True),
        "Volume": st.column_config.NumberColumn(disabled=False)
    })
    st.session_state['reco_bd'] = result_df
    st.info("ℹ️ Edit the above values and click **Apply** to produce the estimated scores after cleanup")

    df_original_result = None
        
    if st.button('Apply Changes'):
        st.divider()
        st.subheader("🧮 Estimated Demographic Scores After Cleanup:")
        st.text("")
        st.write("**Original Demographic Scores:**")
        bar = st.progress(random_integers[0], text="Reconstructing original demographic scores. Please wait.")
        time.sleep(1)

        original_df = st.session_state['reco_bd']
        num_records = int(original_df.loc[original_df['Statistic'] == 'Number of Records', 'Volume'].values[0])
        address_auto = float(original_df.loc[original_df['Statistic'] == 'Address - Auto Update (Final Suggestion)', 'Volume'].values[0])
        address_manual = float(original_df.loc[original_df['Statistic'] == 'Address - Manual Review (Final Suggestion)', 'Volume'].values[0])
        address_good = float(original_df.loc[original_df['Statistic'] == 'Address - Good (Final Suggestion)', 'Volume'].values[0])
        phone_auto = float(original_df.loc[original_df['Statistic'] == 'Phone - Auto Update (Final Suggestion)', 'Volume'].values[0])
        phone_manual = float(original_df.loc[original_df['Statistic'] == 'Phone - Manual Review (Final Suggestion)', 'Volume'].values[0])
        phone_good = float(original_df.loc[original_df['Statistic'] == 'Phone - Good (Final Suggestion)', 'Volume'].values[0])
        specialty_auto = float(original_df.loc[original_df['Statistic'] == 'Specialty - Auto Update', 'Volume'].values[0])
        specialty_manual = float(original_df.loc[original_df['Statistic'] == 'Specialty - Manual Review', 'Volume'].values[0])
        specialty_good = float(original_df.loc[original_df['Statistic'] == 'Specialty - Good', 'Volume'].values[0])

        bar.progress(random_integers[1], text="Reconstructing original demographic scores. Please wait.")
        time.sleep(1)

        average_scores_df = run_score_experiment(seed, num_simulations, num_records, address_auto, address_manual, address_good, phone_auto, phone_manual, phone_good, specialty_auto, specialty_manual, specialty_good)
        st.session_state['average_scores_df'] = average_scores_df

        bar.progress(random_integers[2], text="Reconstructing original demographic scores. Please wait.")
        time.sleep(1)

        if 'average_scores_df' in st.session_state:
            data = st.session_state['average_scores_df'].round(1).to_dict(orient='records')
            bar.progress(random_integers[3], text="Reconstructing original demographic scores. Please wait.")
            time.sleep(1)
            bar.progress(100)
            bar.empty()
            df_formatted = st.session_state['average_scores_df'].map(format_floats)
            df_original_result = df_formatted
            col1, col2, col3 = st.columns([1, 5, 1])
            with col2:
                st.dataframe(df_formatted, hide_index=True)
        
        st.divider()
        st.write("**Demographic Scores After Updates:**")

        with st.status("Computing Scores After Data Cleanup...", expanded=True) as status:
            st.write("Searching for data...")
            time.sleep(2)
            st.write("Reading the data...")
            time.sleep(1)

            st.session_state['reco_bd'] = edited_df
            edited_df = st.session_state['reco_bd']
            num_records = int(edited_df.loc[edited_df['Statistic'] == 'Number of Records', 'Volume'].values[0])
            address_auto = float(edited_df.loc[edited_df['Statistic'] == 'Address - Auto Update (Final Suggestion)', 'Volume'].values[0])
            address_manual = float(edited_df.loc[edited_df['Statistic'] == 'Address - Manual Review (Final Suggestion)', 'Volume'].values[0])
            address_good = float(edited_df.loc[edited_df['Statistic'] == 'Address - Good (Final Suggestion)', 'Volume'].values[0])
            phone_auto = float(edited_df.loc[edited_df['Statistic'] == 'Phone - Auto Update (Final Suggestion)', 'Volume'].values[0])
            phone_manual = float(edited_df.loc[edited_df['Statistic'] == 'Phone - Manual Review (Final Suggestion)', 'Volume'].values[0])
            phone_good = float(edited_df.loc[edited_df['Statistic'] == 'Phone - Good (Final Suggestion)', 'Volume'].values[0])
            specialty_auto = float(edited_df.loc[edited_df['Statistic'] == 'Specialty - Auto Update', 'Volume'].values[0])
            specialty_manual = float(edited_df.loc[edited_df['Statistic'] == 'Specialty - Manual Review', 'Volume'].values[0])
            specialty_good = float(edited_df.loc[edited_df['Statistic'] == 'Specialty - Good', 'Volume'].values[0])

            st.write(f"Running {num_simulations} simulations...")
            time.sleep(1)
            average_scores_df = run_score_experiment(seed, num_simulations, num_records, address_auto, address_manual, address_good, phone_auto, phone_manual, phone_good, specialty_auto, specialty_manual, specialty_good)
            st.session_state['average_scores_df'] = average_scores_df
            st.write(f"Estimating the scores...")
            time.sleep(1)

            if 'average_scores_df' in st.session_state:
                data = st.session_state['average_scores_df'].round(1).to_dict(orient='records')
                df_formatted = st.session_state['average_scores_df'].map(format_floats)

            status.update(
                label="Process complete!", state="complete", expanded=False
            )
        
        a1, a2 = float(df_original_result['Address Score'].iloc[0]), float(df_formatted['Address Score'].iloc[0])
        p1, p2 = float(df_original_result['Phone Score'].iloc[0]), float(df_formatted['Phone Score'].iloc[0])
        s1, s2 = float(df_original_result['Specialty Score'].iloc[0]), float(df_formatted['Specialty Score'].iloc[0])
        d1, d2 = float(df_original_result['Overall Demographic Score'].iloc[0]), float(df_formatted['Overall Demographic Score'].iloc[0])
        actual_d1 = tab1_df['CURRENT_DEMOGRAPHIC_SCORE'].iloc[0]
        absolute_difference = round(abs(actual_d1 - d1), 1)
        
        if (absolute_difference > margin_error):
            variance_flag = 1

        st.text("")
        messages = []
        a_delta = round(a2 - a1, 1)
        p_delta = round(p2 - p1, 1)
        s_delta = round(s2 - s1, 1)
        d_delta = round(d2 - d1, 1)

        col1, col2, col3, col4 = st.columns(4)
        if (a_delta == 0):
            col1.metric("Address Score", a2, a_delta, delta_color="off")
        else:
            col1.metric("Address Score", a2, a_delta)
        
        if (p_delta == 0):
            col2.metric("Phone Score", p2, p_delta, delta_color="off")
        else:
            col2.metric("Phone Score", p2, p_delta)

        if (s_delta == 0):
            col3.metric("Specialty Score", s2, s_delta, delta_color="off")
        else:
            col3.metric("Specialty Score", s2, s_delta)

        if (d_delta == 0):
            col4.metric("Overall Demographic Score", d2, d_delta, delta_color="off")
        else:
            col4.metric("Overall Demographic Score", d2, d_delta)

        if a2 - a1 < 0:
            messages.append(f"- Address Score **decreased by {a1 - a2:.1f}**")
        elif a2 - a1 > 0:
            messages.append(f"- Address Score **increased by {a2 - a1:.1f}**")
        else:
            messages.append(f"- Address Score **remain unchanged**")

        if p2 - p1 < 0:
            messages.append(f"- Phone Score **decreased by {p1 - p2:.1f}**")
        elif p2 - p1 > 0:
            messages.append(f"- Phone Score **increased by {p2 - p1:.1f}**")
        else:
            messages.append(f"- Phone Score **remain unchanged**")

        if s2 - s1 < 0:
            messages.append(f"- Specialty Score **decreased by {s1 - s2:.1f}**")
        elif s2 - s1 > 0:
            messages.append(f"- Specialty Score **increased by {s2 - s1:.1f}**")
        else:
            messages.append(f"- Specialty Score **remain unchanged**")

        if d2 - d1 < 0:
            messages.append(f"- Overall Demographic Score **decreased by {d1 - d2:.1f}**")
            improvement_flag = 1
        elif d2 - d1 > 0:
            messages.append(f"- Overall Demographic Score **increased by {d2 - d1:.1f}**")
        else:
            messages.append(f"- Overall Demographic Score **remain unchanged**")

        final_message = "\n".join(messages)

        st.text("")
        with st.expander(":mag: Understanding the Results", expanded = True):
            if (improvement_flag == 1):
                st.error(final_message)
            else:
                st.success(final_message)

        if (variance_flag == 1):
            with st.expander('⚠️ WARNING'):
                st.warning(f'''The **simulated Demographic Score** (**{d1}**) differs significantly from the **actual Demographic Score** (**{actual_d1}**) on Elixir-P UI. 
              The maximum allowable margin of error is set to **{margin_error}**. 
              Please adjust the attribute-wise scores above to better align with the Elixir-P UI for more accurate predictions.''')
            st.toast('''Process completed with a warning alert!''')
        else:
            st.toast('''Process completed successfully!''')
        
        


with tab2:
    st.info("The counts represent unique **NPI-Address** entries")
    on = st.toggle("Turn ON to change the granularity to RLTD_PADRS_KEY")

    if on:
        pass
