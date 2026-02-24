import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import numpy as np
import requests
import re
import sys
import os

# MUST BE THE FIRST STREAMLIT COMMAND!
st.set_page_config(page_title="Israel Crime Stats", layout="wide")

# ==========================================
# STATIC DATA & CONFIG
# ==========================================
API_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

CITY_COORDINATES = {
    'ירושלים': [31.7683, 35.2137], 'תל אביב - יפו': [32.0853, 34.7818], 'תל אביב-יפו': [32.0853, 34.7818],
    'חיפה': [32.7940, 34.9896], 'ראשון לציון': [31.9730, 34.7925], 'פתח תקווה': [32.0840, 34.8878],
    'אשדוד': [31.8014, 34.6435], 'נתניה': [32.3215, 34.8532], 'באר שבע': [31.2518, 34.7913],
    'בני ברק': [32.0849, 34.8352], 'חולון': [32.0158, 34.7874], 'רמת גן': [32.0684, 34.8248],
    'אשקלון': [31.6693, 34.5715], 'רחובות': [31.8928, 34.8113], 'בת ים': [32.0162, 34.7502],
    'בית שמש': [31.7470, 34.9881], 'כפר סבא': [32.1750, 34.9068], 'הרצליה': [32.1663, 34.8435],
    'חדרה': [32.4340, 34.9207], 'מודיעין-מכבים-רעות': [31.8903, 35.0104], 'לוד': [31.9525, 34.8967],
    'רמלה': [31.9292, 34.8736], 'נצרת': [32.7019, 35.2971], 'רעננה': [32.1848, 34.8713],
    'עכו': [32.9108, 35.0818], 'טבריה': [32.7944, 35.5312], 'אילת': [29.5581, 34.9482],
    'אום אל-פחם': [32.5193, 35.1507], 'רהט': [31.3915, 34.7628], 'הוד השרון': [32.1564, 34.8954],
    'גבעתיים': [32.0715, 34.8089], 'נהרייה': [33.0036, 35.0925], 'נהריה': [33.0036, 35.0925],
    'קריית גת': [31.6111, 34.7685], 'קרית גת': [31.6111, 34.7685], 'קרית אתא': [32.8023, 35.1018],
    'עפולה': [32.6105, 35.2870], 'מודיעין עילית': [31.9304, 35.0381], 'כרמיאל': [32.9190, 35.2951],
    'טייבה': [32.2662, 35.0104], 'נס ציונה': [31.9299, 34.7981], 'קרית מוצקין': [32.8364, 35.0746],
    'ביתר עילית': [31.6961, 35.1118], 'אלעד': [32.0520, 34.9515], 'ראש העין': [32.0956, 34.9566],
    'סחנין': [32.8596, 35.2985], 'יהוד-מונוסון': [32.0319, 34.8906], 'רמת השרון': [32.1481, 34.8385],
    'שפרעם': [32.8058, 35.1702], 'טמרה': [32.8532, 35.1979], 'נתיבות': [31.4172, 34.5878],
    'מגדל העמק': [32.6732, 35.2415], 'אופקים': [31.3129, 34.6186], 'קרית ים': [32.8422, 35.0685],
    'דימונה': [31.0667, 35.0317], 'יבנה': [31.8745, 34.7405], 'טירת כרמל': [32.7667, 34.9667],
    'צפת': [32.9646, 35.4960], 'מעלה אדומים': [31.7770, 35.2995], 'קרית ביאליק': [32.8375, 35.0833],
    'אור יהודה': [32.0292, 34.8550], 'קרית אונו': [32.0628, 34.8572], 'קרית מלאכי': [31.7317, 34.7431],
    'גסר א-זרקא': [32.5367, 34.9125], 'ג\'סר א-זרקא': [32.5367, 34.9125], 'ערד': [31.2612, 35.2144],
    'כפר יונה': [32.3168, 34.9351], 'קרית שמונה': [33.2073, 35.5721], 'נמל תעופה בן-גוריון': [32.0006, 34.8708],
    'חריש': [32.4633, 35.0450], 'נשר': [32.7758, 35.0428], 'מבשרת ציון': [31.7958, 35.1561],
    'גן יבנה': [31.7850, 34.7170], 'זכרון יעקב': [32.5714, 34.9522], 'קדימה-צורן': [32.2792, 34.9150],
    'גדרה': [31.8125, 34.7780],
}

# ==========================================
# DATA PIPELINE FUNCTIONS (ONLY RUNS LOCALLY)
# ==========================================

def load_crime_df():
    resources = {
        "2025": "e311b6a1-be5a-4a82-8298-f3afbee07b6b", "2024": "5fc13c50-b6f3-4712-b831-a75e0f91a17e",
        "2023": "32aacfc9-3524-4fba-a282-3af052380244", "2022": "a59f3e9e-a7fe-4375-97d0-76cea68382c1",
        "2021": "3f71fd16-25b8-4cfe-8661-e6199db3eb12", "2020": "520597e3-6003-4247-9634-0ae85434b971"
    }
    all_dfs = []
    for year, resource_id in resources.items():
        print(f"loading {year}...")
        url = f"https://data.gov.il/api/3/action/datastore_search"
        limit, offset = 50000, 0
        rows = []
        while True:
            resp = requests.get(url, params={"resource_id": resource_id, "limit": limit, "offset": offset}, headers=API_HEADERS)
            data = resp.json()
            if 'result' not in data: break
            batch = data["result"]["records"]
            rows.extend(batch)
            if len(batch) < limit: break 
            offset += limit
        all_dfs.append(pd.DataFrame(rows))
    return pd.concat(all_dfs, ignore_index=True).drop_duplicates()

def clean_text_columns_regex(df, columns_to_clean):
    df_cleaned = df.copy()
    pattern = r'[^\w\s]|_'
    for col in columns_to_clean:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].apply(lambda x: re.sub(pattern, '', str(x)).strip() if pd.notna(x) else x)
    return df_cleaned

def impute_missing_names_from_codes(df, code_name_pairs):
    df_out = df.copy()
    for code_col, name_col in code_name_pairs:
        if code_col not in df_out.columns or name_col not in df_out.columns: continue
        valid_rows = df_out.dropna(subset=[code_col, name_col])
        if valid_rows.empty: continue
        mapping_dict = dict(zip(valid_rows[code_col], valid_rows[name_col]))
        mask = df_out[name_col].isna() & df_out[code_col].notna()
        df_out.loc[mask, name_col] = df_out.loc[mask, code_col].map(mapping_dict)
    return df_out

def impute_fields_from_station_text(df, station_col, target_cols):
    df_out = df.copy()
    if station_col not in df_out.columns: return df_out
    FORBIDDEN_VALUES = {'', ' ', 'מקום אחר', 'other', 'Other', 'לא ידוע', 'nan', 'None'}
    for target_col in target_cols:
        if target_col not in df_out.columns or df_out[target_col].isna().sum() == 0: continue
        valid_candidates = sorted([v for v in df_out[target_col].dropna().unique() if str(v).strip() not in FORBIDDEN_VALUES], key=lambda x: len(str(x)), reverse=True)
        mask = df_out[target_col].isna() & df_out[station_col].notna()
        mapping = {}
        for station in df_out.loc[mask, station_col].unique():
            for candidate in valid_candidates:
                if str(candidate) in str(station):
                    mapping[station] = candidate
                    break 
        if mapping:
            rows_to_fill = mask & df_out[station_col].isin(mapping.keys())
            df_out.loc[rows_to_fill, target_col] = df_out.loc[rows_to_fill, station_col].map(mapping)
    return df_out

def impute_parent_from_child(df, child_col, parent_col):
    df_out = df.copy()
    if child_col not in df_out.columns or parent_col not in df_out.columns or df_out[parent_col].isna().sum() == 0: return df_out
    valid_relations = df_out.dropna(subset=[child_col, parent_col])[[child_col, parent_col]].drop_duplicates()
    valid_relations = valid_relations[~valid_relations[parent_col].astype(str).isin(['', ' ', 'מקום אחר', 'nan'])]
    ambiguous_children = valid_relations[valid_relations.duplicated(subset=[child_col], keep=False)][child_col].unique()
    valid_relations = valid_relations[~valid_relations[child_col].isin(ambiguous_children)]
    mask = df_out[parent_col].isna() & df_out[child_col].notna()
    df_out.loc[mask, parent_col] = df_out.loc[mask, child_col].map(dict(zip(valid_relations[child_col], valid_relations[parent_col])))
    return df_out

def process_and_summarize_crime_data(df):
    impute_pairs = [
        ('municipalKod', 'municipalName'), ('YeshuvKod', 'Yeshuv'),
        ('PoliceDistrictKod', 'PoliceDistrict'), ('PoliceMerhavKod', 'PoliceMerhav'),
        ('StatisticAreaKod', 'StatisticArea'), ('StatisticGroupKod', 'StatisticGroup'),
        ('StatisticTypeKod', 'StatisticType')
    ]
    text_cols = [pair[1] for pair in impute_pairs] + ['PoliceStation'] 
    df_processed = clean_text_columns_regex(df, [c for c in text_cols if c in df.columns])
    df_processed = impute_missing_names_from_codes(df_processed, impute_pairs)
    df_processed = impute_parent_from_child(df_processed, 'PoliceStation', 'Yeshuv')
    df_processed = impute_fields_from_station_text(df_processed, 'PoliceStation', ['Yeshuv', 'PoliceMerhav', 'PoliceDistrict', 'municipalName'])
    for child, parent in [('Yeshuv', 'PoliceMerhav'), ('Yeshuv', 'municipalName'), ('PoliceMerhav', 'PoliceDistrict')]:
        df_processed = impute_parent_from_child(df_processed, child, parent)

    group_cols = ['Year', 'Quarter', 'Yeshuv', 'YeshuvKod', 'StatisticGroup', 'StatisticType']
    valid_group_cols = [c for c in group_cols if c in df_processed.columns]
    
    df_for_agg = df_processed.copy()
    for col in valid_group_cols: df_for_agg[col] = df_for_agg[col].fillna('Missing')
    
    summary_df = df_for_agg.groupby(valid_group_cols)['FictiveIDNumber'].count().reset_index()
    summary_df.rename(columns={'FictiveIDNumber': 'EventCount'}, inplace=True)
    return summary_df

def fetch_population_data():
    resources = {
        2019: '990ae78e-2dae-4a15-a13b-0b5dcc56056c', 2020: '2d218594-73e3-40de-b36b-23b22f0a2627',
        2021: '95435941-d7e5-46c6-876a-761a74a5928d', 2022: '199b15db-3bcb-470e-ba03-73364737e352',
        2023: 'd47a54ff-87f0-44b3-b33a-f284c0c38e5a'
    }
    pop_dfs = []
    field_mapping = {
        'Yeshuv_Code': 'סמל יישוב', 'Yeshuv_Name': 'שם יישוב', 'Religion_Code': 'דת יישוב', 
        'Total_Population': 'סך הכל אוכלוסייה', 'Total_Israelis': 'סך הכל ישראלים', 
        'Jews_and_Others': 'יהודים ואחרים', 'Arabs': 'ערבים'
    }
    for year, resource_id in resources.items():
        try:
            resp = requests.get("https://data.gov.il/api/3/action/datastore_search", params={'resource_id': resource_id, 'limit': 5000}, headers=API_HEADERS).json()
            if resp.get('success'):
                df_temp = pd.DataFrame(resp['result']['records'])
                found_cols = {}
                for eng_key, heb_search in field_mapping.items():
                    match = next((c for c in df_temp.columns if heb_search in c and ('אנגלית' not in c if eng_key == 'Yeshuv_Name' else True)), None)
                    if match: found_cols[eng_key] = match
                if 'Yeshuv_Code' in found_cols:
                    df_clean = df_temp[list(found_cols.values())].rename(columns={v: k for k, v in found_cols.items()}).copy()
                    df_clean['Year'] = year
                    pop_dfs.append(df_clean)
        except Exception: pass
    return pd.concat(pop_dfs, ignore_index=True) if pop_dfs else pd.DataFrame()

def extrapolate_quarters(pop_df, years_of_interest):
    if pop_df.empty: return pop_df
    pop_df['Yeshuv_Code'] = pd.to_numeric(pop_df['Yeshuv_Code'], errors='coerce')
    for col in ['Total_Population', 'Total_Israelis', 'Jews_and_Others', 'Arabs', 'Religion_Code']:
        if col in pop_df.columns:
            if pop_df[col].dtype == object: pop_df[col] = pop_df[col].astype(str).str.replace(',', '')
            pop_df[col] = pd.to_numeric(pop_df[col], errors='coerce')
            
    expanded_data = []
    for yeshuv_code, group in pop_df.dropna(subset=['Yeshuv_Code']).groupby('Yeshuv_Code'):
        group = group.sort_values('Year')
        yeshuv_name = group['Yeshuv_Name'].iloc[0] if 'Yeshuv_Name' in group.columns else ''
        for year in years_of_interest:
            curr = group[group['Year'] == year]
            if curr.empty: continue
            curr_pop = curr['Total_Population'].values[0] if pd.notna(curr['Total_Population'].values[0]) else 0
            prev = group[group['Year'] == year - 1]
            quarters = [curr_pop]*4
            if not prev.empty and pd.notna(prev['Total_Population'].values[0]):
                prev_pop = prev['Total_Population'].values[0]
                q_growth = (curr_pop - prev_pop) / 4
                quarters = [prev_pop + q_growth, prev_pop + q_growth*2, prev_pop + q_growth*3, curr_pop]
            
            for q_idx, q_name in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
                expanded_data.append({
                    'Yeshuv_Code': yeshuv_code, 'Yeshuv_Name': yeshuv_name, 'Year': year, 'Quarter': q_name,
                    'Total_Population': int(quarters[q_idx]) if pd.notna(quarters[q_idx]) else 0, 
                    'Religion_Code': curr['Religion_Code'].values[0] if 'Religion_Code' in curr else np.nan,
                    'Jews_and_Others': curr['Jews_and_Others'].values[0] if 'Jews_and_Others' in curr else np.nan,
                    'Arabs': curr['Arabs'].values[0] if 'Arabs' in curr else np.nan
                })
    return pd.DataFrame(expanded_data)

def join_crime_population(crime_df_agg, pop_quarterly_df):
    crime_df_clean = crime_df_agg.dropna(subset=['Yeshuv', 'Year']).copy()
    crime_df_clean['Join_Key_Code'] = pd.to_numeric(crime_df_clean['YeshuvKod'], errors='coerce')
    available_cols = [c for c in ['Yeshuv_Code', 'Year', 'Quarter', 'Total_Population', 'Religion_Code', 'Jews_and_Others', 'Arabs'] if c in pop_quarterly_df.columns]
    merged = pd.merge(crime_df_clean, pop_quarterly_df[available_cols], left_on=['Join_Key_Code', 'Year', 'Quarter'], right_on=['Yeshuv_Code', 'Year', 'Quarter'], how='left')
    return merged.drop(columns=['Join_Key_Code', 'Yeshuv_Code'], errors='ignore')

def get_chained_quarterly_cpi(start_year, end_year):
    try:
        resp = requests.get("https://api.cbs.gov.il/index/data/price", params={"id": 120010, "startPeriod": f"01-{start_year}", "endPeriod": f"12-{end_year}", "format": "json", "download": "false"}, headers=API_HEADERS).json()
        if 'month' not in resp or not resp['month']: return None
        df = pd.DataFrame([{'date': f"{o['year']}-{o['month']:02d}-01", 'monthly_percent_change': float(o['percent'])} for o in resp['month'][0].get('date', []) if o.get('percent') is not None])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df['chained_index'] = 100.0
        for i in range(1, len(df)): df.loc[i, 'chained_index'] = df.loc[i-1, 'chained_index'] * (1 + df.loc[i, 'monthly_percent_change'] / 100)
        q_df = df.set_index('date').resample('Q').agg({'chained_index': 'mean', 'monthly_percent_change': 'sum'}).reset_index()
        q_df['quarter_name'] = q_df['date'].dt.to_period('Q')
        return q_df.rename(columns={'chained_index': 'avg_chained_index_points', 'monthly_percent_change': 'total_quarterly_inflation_pct'})[['quarter_name', 'avg_chained_index_points', 'total_quarterly_inflation_pct']]
    except Exception: return None

def run_data_pipeline(final_crime_path, cpi_path):
    print("Initiating Pipeline...")
    crime_df = load_crime_df()
    crime_df_agg = process_and_summarize_crime_data(crime_df)
    pop_quarterly = extrapolate_quarters(fetch_population_data(), [2020, 2021, 2022, 2023])
    if not pop_quarterly.empty:
        join_crime_population(crime_df_agg, pop_quarterly).to_csv(final_crime_path, index=False, encoding='utf-8-sig', compression='gzip')
    cpi_df = get_chained_quarterly_cpi(2020, 2025)
    if cpi_df is not None: cpi_df.to_csv(cpi_path, index=False, encoding='utf-8-sig')
    print("Pipeline Complete!")

# ==========================================
# DASHBOARD FUNCTIONS
# ==========================================

def determine_majority_religion(df):
    potential_cols = ['Jews_and_Others', 'Arabs', 'יהודים ואחרים', 'ערבים']
    existing_cols = [col for col in potential_cols if col in df.columns]
    if len(existing_cols) >= 2:
        temp_df = df[existing_cols].replace({',': ''}, regex=True).apply(pd.to_numeric, errors='coerce').fillna(0)
        majority = temp_df.idxmax(axis=1)
        name_mapping = {'יהודים ואחרים': 'Jews & Others', 'Jews_and_Others': 'Jews & Others', 'ערבים': 'Arabs', 'Arabs': 'Arabs'}
        df['Majority_Religion'] = majority.map(name_mapping).fillna('Data Not Available')
    else:
        df['Majority_Religion'] = 'Data Not Available'
    return df

@st.cache_data(show_spinner=False)
def load_dashboard_data(crime_path, cpi_path):
    # This prevents the cloud from trying to run the pipeline if files are missing
    if not os.path.exists(crime_path):
        return None, None
    df = pd.read_csv(crime_path, low_memory=False, compression='gzip')
    cpi_df = pd.read_csv(cpi_path) if os.path.exists(cpi_path) else None
    return df, cpi_df

def visualize_crime_rates_streamlit(merged_df, cpi_df=None):
    st.title("🛡️ Israel Crime Analysis")
    DISTINCT_COLORS = px.colors.qualitative.Pastel + px.colors.qualitative.Set3 + px.colors.qualitative.Pastel1 + px.colors.qualitative.Pastel2

    merged_df.columns = merged_df.columns.str.strip()
    df_clean = merged_df[merged_df['Year'].isin([2020, 2021, 2022, 2023])].copy()
    df_clean = df_clean[df_clean['StatisticGroup'] != 'שגיאת הזנה']
    df_clean['EventCount'] = pd.to_numeric(df_clean['EventCount'], errors='coerce').fillna(0)
    df_clean['Total_Population'] = pd.to_numeric(df_clean['Total_Population'], errors='coerce')
    df_clean = determine_majority_religion(df_clean)

    st.sidebar.header("Global Data Filters")
    st.sidebar.markdown("Changes here affect all visualizations.")

    selected_year = st.sidebar.selectbox("Select Year", ["Average", 2020, 2021, 2022, 2023], index=0)
    quarter_options = ["All Quarters"] + (sorted(df_clean['Quarter'].dropna().unique().tolist()) if 'Quarter' in df_clean.columns else [])
    selected_quarter = st.sidebar.selectbox("Select Quarter", quarter_options, index=0)

    all_groups = ["All Groups"] + sorted(df_clean['StatisticGroup'].dropna().unique().tolist())
    selected_group = st.sidebar.selectbox("Select Crime Group (Type)", all_groups, index=0)
    
    subtypes = ["All Sub-Types"] + sorted(df_clean[df_clean['StatisticGroup'] == selected_group]['StatisticType'].dropna().unique().tolist() if selected_group != "All Groups" else df_clean['StatisticType'].dropna().unique().tolist())
    selected_subtype = st.sidebar.selectbox("Select Crime Sub-Type", subtypes, index=0)

    all_religions = ["All Populations"] + sorted(df_clean['Majority_Religion'].dropna().unique().tolist())
    selected_religion = st.sidebar.selectbox("Select Population Majority", all_religions, index=0)

    df_filtered = df_clean.copy()
    if selected_quarter != "All Quarters": df_filtered = df_filtered[df_filtered['Quarter'] == selected_quarter]
    if selected_group != "All Groups": df_filtered = df_filtered[df_filtered['StatisticGroup'] == selected_group]
    if selected_subtype != "All Sub-Types": df_filtered = df_filtered[df_filtered['StatisticType'] == selected_subtype]
    if selected_religion != "All Populations": df_filtered = df_filtered[df_filtered['Majority_Religion'] == selected_religion]

    agg_yearly = df_filtered.groupby(['Year', 'Yeshuv', 'Majority_Religion']).agg({'EventCount': 'sum', 'Total_Population': 'max'}).reset_index()
    agg_yearly['Crime_Rate'] = (agg_yearly['EventCount'] / agg_yearly['Total_Population']) * 1000
    agg_yearly['Crime_Percentage'] = (agg_yearly['EventCount'] / agg_yearly['Total_Population']) * 100
    agg_yearly = agg_yearly[agg_yearly['Total_Population'] > 0].dropna(subset=['Crime_Rate'])

    agg_avg = agg_yearly.groupby(['Yeshuv', 'Majority_Religion']).agg({'EventCount': 'mean', 'Total_Population': 'mean', 'Crime_Rate': 'mean', 'Crime_Percentage': 'mean'}).reset_index()
    agg_avg['Year'] = 'Average'

    current_view_df = agg_avg.copy() if selected_year == "Average" else agg_yearly[agg_yearly['Year'] == selected_year].copy()
    period_label = f"Average (2020-2023)" if selected_year == "Average" else str(selected_year)
    if selected_quarter != "All Quarters": period_label += f" | {selected_quarter}"

    total_crimes = int(current_view_df['EventCount'].sum()) if not current_view_df.empty else 0
    avg_rate = current_view_df['Crime_Rate'].mean() if not current_view_df.empty else 0.0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"Total Crimes ({selected_year})", f"{total_crimes:,}")
    col2.metric("Avg National Crime Rate*", f"{avg_rate:.2f}" if not np.isnan(avg_rate) else "0.00")
    col3.metric("Selected Period", period_label)
    col4.metric("Selected Population", selected_religion)
    st.caption("* Crime Rate = crime incidents per 1,000 residents")
    st.markdown("---")

    # 1. TREEMAP
    st.header("1. 📊 National Crime Distribution - Most crimes are 'Against Public Order', threats specifically with 14%")
    treemap_agg = df_filtered.groupby(['Year', 'StatisticGroup', 'StatisticType'])['EventCount'].sum().reset_index()
    if selected_year == "Average": treemap_agg = treemap_agg.groupby(['StatisticGroup', 'StatisticType'])['EventCount'].mean().reset_index()
    else: treemap_agg = treemap_agg[treemap_agg['Year'] == selected_year]

    if not treemap_agg.empty and treemap_agg['EventCount'].sum() > 0:
        treemap_agg['Share'] = (treemap_agg['EventCount'] / treemap_agg['EventCount'].sum()) * 100
        fig_tree = px.treemap(treemap_agg, path=[px.Constant("All Crimes"), 'StatisticGroup', 'StatisticType'], values='EventCount', color='StatisticGroup', title=f"Crime Breakdown - {period_label}", color_discrete_sequence=DISTINCT_COLORS)
        fig_tree.update_traces(textinfo="label+value+percent root", hovertemplate="<b>%{label}</b><br>Count: %{value:,.0f}<br>Share of Total Crime: %{percentRoot:.1%}<br>Share of Category: %{percentParent:.1%}<extra></extra>")
        fig_tree.update_layout(margin=dict(t=50, l=25, r=25, b=25), height=600)
        st.plotly_chart(fig_tree, use_container_width=True)
    else: st.info("No data available for the selected filter.")
    st.markdown("---")

    # 2. OVERALL TREND
    st.header("2. 📉 Overall Crime Rate Trend Over Time - 2023 had even lower crime rate than COVID lockdown in 2021")
    st.markdown("Examines whether the total volume of crime (normalized per 1,000 residents) is changing. Note the volatility of the data: it is not a clean trend, but more like strains by each year, as this is raw data without processing.")
    if 'Quarter' in df_filtered.columns:
        trend_base = df_filtered.groupby(['Year', 'Quarter', 'Yeshuv']).agg({'EventCount': 'sum', 'Total_Population': 'max'}).reset_index()
        trend_base['YearQuarter'] = trend_base['Year'].astype(str) + "-" + trend_base['Quarter']
        nat_trend = trend_base.groupby('YearQuarter').agg({'EventCount': 'sum', 'Total_Population': 'sum'}).reset_index()
        nat_trend = nat_trend[nat_trend['Total_Population'] > 0]
        nat_trend['National_Crime_Rate'] = (nat_trend['EventCount'] / nat_trend['Total_Population']) * 1000
        fig_nat_trend = px.line(nat_trend, x='YearQuarter', y='National_Crime_Rate', markers=True, text=nat_trend['National_Crime_Rate'].apply(lambda x: f"{x:.2f}"), title="Normalized Crime Rate Over Time (per 1,000 Residents)", labels={'YearQuarter': 'Quarter', 'National_Crime_Rate': 'Crime Rate (per 1k)'})
        fig_nat_trend.update_traces(textposition="top center", line_color='#1f77b4', marker=dict(size=8))
        st.plotly_chart(fig_nat_trend, use_container_width=True)
    st.markdown("---")

    # 3. DISTRIBUTION OVER TIME
    st.header("3. 📈 Crime Volume & Distribution Over Time - the share of crimes against property is increasing with time")
    st.markdown("Displays the absolute number of crimes per quarter, with internal segments representing the percentage of each crime type.")
    if 'Quarter' in df_filtered.columns:
        q1_df = df_filtered.groupby(['Year', 'Quarter', 'StatisticGroup'])['EventCount'].sum().reset_index()
        q1_df['YearQuarter'] = q1_df['Year'].astype(str) + "-" + q1_df['Quarter']
        q1_df['Percent'] = (q1_df['EventCount'] / q1_df.groupby('YearQuarter')['EventCount'].transform('sum')) * 100
        cat_order = q1_df.groupby('StatisticGroup')['EventCount'].sum().sort_values(ascending=False).index.tolist()
        fig_q1 = px.bar(q1_df, x='YearQuarter', y='EventCount', color='StatisticGroup', title="Absolute Crime Volume with Relative Distribution", text=q1_df['Percent'].apply(lambda x: f"{x:.0f}%" if x >= 3 else ""), category_orders={'StatisticGroup': cat_order}, color_discrete_sequence=DISTINCT_COLORS)
        fig_q1.update_traces(textposition='inside', textfont_size=12)
        st.plotly_chart(fig_q1, use_container_width=True)
    st.markdown("---")

    # 4. CPI
    st.header("4. 💰 Correlation Observed Between CPI and Share of Property Crimes")
    if cpi_df is not None and not cpi_df.empty:
        cpi_df.columns = cpi_df.columns.str.strip()
        q4_df = df_clean.groupby(['Year', 'Quarter', 'StatisticGroup'])['EventCount'].sum().reset_index()
        q4_df['YearQuarter'] = q4_df['Year'].astype(str) + q4_df['Quarter']
        q4_pivot = q4_df.pivot(index='YearQuarter', columns='StatisticGroup', values='EventCount').fillna(0)
        q4_pct = q4_pivot.div(q4_pivot.sum(axis=1), axis=0) * 100
        prop_cols = [c for c in q4_pct.columns if 'רכוש' in str(c) or 'Property' in str(c)]
        
        if prop_cols and 'quarter_name' in cpi_df.columns and 'avg_chained_index_points' in cpi_df.columns:
            col_name = prop_cols[0]
            merged_cpi = q4_pct[[col_name]].merge(cpi_df, left_index=True, right_on=cpi_df['quarter_name'].astype(str).str.strip()).dropna(subset=['avg_chained_index_points', col_name]).sort_values('avg_chained_index_points')
            x_vals, y_vals = merged_cpi['avg_chained_index_points'], merged_cpi[col_name]
            
            fig_q4 = go.Figure()
            fig_q4.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='markers', name='Property Crimes', marker=dict(color='royalblue', size=10)))
            
            try:
                import statsmodels.api as sm
                X_sm = sm.add_constant(x_vals)
                pred_df = sm.OLS(y_vals, X_sm).fit().get_prediction(X_sm).summary_frame(alpha=0.05)
                fig_q4.add_trace(go.Scatter(x=x_vals, y=pred_df['mean'], mode='lines', name='Trend', line=dict(color='royalblue', width=2)))
                fig_q4.add_trace(go.Scatter(x=x_vals.tolist() + x_vals.tolist()[::-1], y=pred_df['mean_ci_upper'].tolist() + pred_df['mean_ci_lower'].tolist()[::-1], fill='toself', fillcolor='rgba(65, 105, 225, 0.2)', line=dict(color='rgba(255,255,255,0)'), hoverinfo='skip', showlegend=False, name='95% CI'))
            except ImportError: st.warning("Trendline hidden because 'statsmodels' is missing.")
            
            fig_q4.update_layout(title="Consumer Price Index (CPI) vs. Percentage of Property Crimes", xaxis_title="CPI Points", yaxis_title="Percentage of Property Crimes (%)", hovermode="x unified")
            st.plotly_chart(fig_q4, use_container_width=True)
    else: st.warning("Quarterly CPI data not found.")
    st.markdown("---")

    # 5. MAP
    st.header("5. 🗺️ Crime rates are higher in suburbs compared to the center")
    map_data = current_view_df.copy()
    map_data['coords'] = map_data['Yeshuv'].apply(lambda x: CITY_COORDINATES.get(x, [None, None]))
    map_data[['lat', 'lon']] = pd.DataFrame(map_data['coords'].tolist(), index=map_data.index)
    map_plot_df = map_data.dropna(subset=['lat', 'lon']).copy()
    
    if not map_plot_df.empty:
        for c in ['Crime_Rate', 'Crime_Percentage', 'EventCount', 'Total_Population']: map_plot_df[f'fmt_{c}'] = map_plot_df[c].round(0).astype(int).astype(str) + ('%' if c == 'Crime_Percentage' else '')
        map_plot_df['color'] = map_plot_df['Crime_Rate'].apply(lambda r: [224,243,252,200] if r<10 else [158,202,225,200] if r<20 else [66,146,198,200] if r<30 else [8,81,156,200] if r<40 else [8,48,107,220])
        st.markdown("<div style='display:flex;gap:15px;margin-bottom:10px'><span style='font-weight:bold'>Crime Rate:</span><span style='color:#e0f3fc'>■ <10</span><span style='color:#9ecae1'>■ 10-20</span><span style='color:#4292c6'>■ 20-30</span><span style='color:#08519c'>■ 30-40</span><span style='color:#08306b'>■ >40</span></div>", unsafe_allow_html=True)
        st.pydeck_chart(pdk.Deck(layers=[pdk.Layer("ScatterplotLayer", map_plot_df, get_position='[lon, lat]', get_color='color', get_radius=1500, radius_min_pixels=5, radius_max_pixels=40, pickable=True)], initial_view_state=pdk.ViewState(latitude=31.5, longitude=34.8, zoom=6.5), tooltip={"html": "<b>{Yeshuv}</b><br/>Pop: {Majority_Religion}<br/>Rate: {fmt_Crime_Rate}<br/>Events: {fmt_EventCount}", "style": {"backgroundColor": "#111", "color": "white"}}))
    st.markdown("---")

    # 6. TOP 10 CITIES
    st.header("6. 📍 Top 10 Cities by Crime Rate - 4 out of 10 cities are Arab majority")
    min_pop = st.slider("Minimum City Population Filter", 1000, 50000, 5000, step=1000)
    top_20 = current_view_df[current_view_df['Total_Population'] >= min_pop].sort_values('Crime_Rate', ascending=False).head(10)
    if not top_20.empty:
        fig_bar = px.bar(top_20, x='Crime_Rate', y='Yeshuv', orientation='h', title=f"Top 10 Cities by Crime Rate ({period_label})", color='Crime_Rate', color_continuous_scale='Blues', hover_data=['Majority_Religion', 'Total_Population'])
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, height=500)
        st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown("---")

    # 7. BOXPLOT
    st.header("7. 📊 Correlation Observed Between Locality Demographic and Dominant Crime Profile")
    df_demo = df_filtered[df_filtered['Majority_Religion'].isin(['Jews & Others', 'Arabs'])].copy()
    if not df_demo.empty:
        city_comp = df_demo.groupby(['Yeshuv', 'Majority_Religion', 'StatisticGroup'])['EventCount'].sum().reset_index().merge(df_demo.groupby('Yeshuv')['EventCount'].sum().reset_index().rename(columns={'EventCount': 'CityTotal'}), on='Yeshuv')
        city_comp['Percent'] = (city_comp['EventCount'] / city_comp['CityTotal']) * 100
        st.info(f"💡 **Analysis Scope:** {city_comp[['Yeshuv', 'Majority_Religion']].drop_duplicates()['Majority_Religion'].value_counts().get('Jews & Others', 0)} Jewish localities, {city_comp[['Yeshuv', 'Majority_Religion']].drop_duplicates()['Majority_Religion'].value_counts().get('Arabs', 0)} Arab localities.")
        fig_demo = px.box(city_comp, x='StatisticGroup', y='Percent', color='Majority_Religion', title="Demographic Variance in Crime Type Distribution", category_orders={'StatisticGroup': city_comp.groupby('StatisticGroup')['Percent'].median().sort_values(ascending=False).index.tolist()}, color_discrete_map={'Jews & Others': px.colors.qualitative.Pastel[1], 'Arabs': px.colors.qualitative.Pastel[4]})
        fig_demo.update_layout(xaxis_tickangle=-45, height=600)
        st.plotly_chart(fig_demo, use_container_width=True)
    st.markdown("---")

    # 8. PERIOD IMPACT
    st.header("8. 🦠⚔️ Crime Distribution Across National Periods - Property crime shares were lower during emergencies compared to routine")
    df_q2 = df_filtered.copy()
    if not df_q2.empty:
        df_q2['Period'] = df_q2.apply(lambda r: "COVID-19 Period" if r['Year'] in [2020, 2021] else ("Iron Swords War" if (r['Year']==2023 and r['Quarter']=='Q4') or r['Year']==2024 else "Routine"), axis=1)
        q2_df = df_q2.groupby(['Period', 'StatisticGroup'])['EventCount'].sum().reset_index()
        q2_df['Percent'] = (q2_df['EventCount'] / q2_df.groupby('Period')['EventCount'].transform('sum')) * 100
        q2_df['Period'] = pd.Categorical(q2_df['Period'], categories=['COVID-19 Period', 'Routine', 'Iron Swords War'], ordered=True)
        fig_q2 = px.line(q2_df.sort_values(['StatisticGroup', 'Period']), x='Period', y='Percent', color='StatisticGroup', markers=True, text=q2_df['Percent'].apply(lambda x: f"{x:.1f}%" if x >= 1 else ""), color_discrete_sequence=DISTINCT_COLORS)
        fig_q2.update_traces(textposition='top center', line=dict(width=3), marker=dict(size=8))
        fig_q2.update_layout(height=800)
        st.plotly_chart(fig_q2, use_container_width=True)
    st.markdown("---")

    # 9. SOCIO-ECONOMIC
    cluster_cols = [c for c in df_clean.columns if 'cluster' in c.lower() or 'socio' in c.lower() or 'אשכול' in c]
    if cluster_cols:
        st.header("9. 🏙️ Correlation Between Socio-Economic Cluster and Crime Rate")
        df_q5 = (agg_avg.copy() if selected_year == "Average" else agg_yearly[agg_yearly['Year'] == selected_year].copy()).merge(df_clean[['Yeshuv', cluster_cols[0]]].dropna().drop_duplicates(subset=['Yeshuv']), on='Yeshuv', how='inner')
        if not df_q5.empty:
            st.plotly_chart(px.scatter(df_q5, x=cluster_cols[0], y='Crime_Rate', size='EventCount', hover_name='Yeshuv', title="Socio-Economic Cluster vs. Crime Rate"), use_container_width=True)

    with st.expander("View Raw Aggregated Data"): st.dataframe(current_view_df.sort_values(['Crime_Rate'], ascending=[False]))

# --- MAIN APP EXECUTION ---
if __name__ == "__main__":
    FINAL_CRIME_PATH = "merged_crime_population_final.csv.gz"
    CPI_PATH = "quarterly_cpi_chained.csv"

    # 1. RUNNING ON STREAMLIT CLOUD
    if st.runtime.exists():
        df, cpi_df = load_dashboard_data(FINAL_CRIME_PATH, CPI_PATH)
        
        if df is None:
            st.error("🚨 Missing Data Files!")
            st.warning("""
            The cloud server does not have the pre-processed data files.
            
            **How to fix this:**
            1. Open your repository on GitHub (`zelignoam/final-project-data-visualization`).
            2. Click **Add file** -> **Upload files**.
            3. Upload these exact two files from your computer:
               * `merged_crime_population_final.csv.gz`
               * `quarterly_cpi_chained.csv`
            4. Commit the changes.
            
            Once the files are there, this page will automatically update!
            """)
            st.stop() # ABSOLUTELY DO NOT TRY TO RUN PIPELINE ON CLOUD
            
        else:
            visualize_crime_rates_streamlit(df, cpi_df)

    # 2. RUNNING LOCALLY ON YOUR MAC
    else:
        print("------------------------------------------------------------------")
        print("⚠️  STREAMLIT NOT DETECTED (Running in Terminal Mode)")
        if not os.path.exists(FINAL_CRIME_PATH) or not os.path.exists(CPI_PATH):
            print(f"Data files missing locally. Initiating automated Data Pipeline to build them...")
            run_data_pipeline(FINAL_CRIME_PATH, CPI_PATH)
        else:
            print(f"Data files exist locally. To view the dashboard, run:")
            print(f"   $ streamlit run {os.path.basename(__file__) if '__file__' in locals() else 'Final_project_streamlit.py'}")
            
            rebuild = input("\nDo you want to RE-RUN the Data Pipeline anyway? (y/n): ").strip().lower()
            if rebuild == 'y':
                run_data_pipeline(FINAL_CRIME_PATH, CPI_PATH)
        print("------------------------------------------------------------------")
