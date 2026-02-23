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
# Fake a browser identity to prevent Gov APIs from blocking cloud servers
API_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

CITY_COORDINATES = {
    'ירושלים': [31.7683, 35.2137],
    'תל אביב - יפו': [32.0853, 34.7818],
    'תל אביב-יפו': [32.0853, 34.7818],
    'חיפה': [32.7940, 34.9896],
    'ראשון לציון': [31.9730, 34.7925],
    'פתח תקווה': [32.0840, 34.8878],
    'אשדוד': [31.8014, 34.6435],
    'נתניה': [32.3215, 34.8532],
    'באר שבע': [31.2518, 34.7913],
    'בני ברק': [32.0849, 34.8352],
    'חולון': [32.0158, 34.7874],
    'רמת גן': [32.0684, 34.8248],
    'אשקלון': [31.6693, 34.5715],
    'רחובות': [31.8928, 34.8113],
    'בת ים': [32.0162, 34.7502],
    'בית שמש': [31.7470, 34.9881],
    'כפר סבא': [32.1750, 34.9068],
    'הרצליה': [32.1663, 34.8435],
    'חדרה': [32.4340, 34.9207],
    'מודיעין-מכבים-רעות': [31.8903, 35.0104],
    'לוד': [31.9525, 34.8967],
    'רמלה': [31.9292, 34.8736],
    'נצרת': [32.7019, 35.2971],
    'רעננה': [32.1848, 34.8713],
    'עכו': [32.9108, 35.0818],
    'טבריה': [32.7944, 35.5312],
    'אילת': [29.5581, 34.9482],
    'אום אל-פחם': [32.5193, 35.1507],
    'רהט': [31.3915, 34.7628],
    'הוד השרון': [32.1564, 34.8954],
    'גבעתיים': [32.0715, 34.8089],
    'נהרייה': [33.0036, 35.0925],
    'נהריה': [33.0036, 35.0925],
    'קריית גת': [31.6111, 34.7685],
    'קרית גת': [31.6111, 34.7685],
    'קרית אתא': [32.8023, 35.1018],
    'עפולה': [32.6105, 35.2870],
    'מודיעין עילית': [31.9304, 35.0381],
    'כרמיאל': [32.9190, 35.2951],
    'טייבה': [32.2662, 35.0104],
    'נס ציונה': [31.9299, 34.7981],
    'קרית מוצקין': [32.8364, 35.0746],
    'ביתר עילית': [31.6961, 35.1118],
    'אלעד': [32.0520, 34.9515],
    'ראש העין': [32.0956, 34.9566],
    'סחנין': [32.8596, 35.2985],
    'יהוד-מונוסון': [32.0319, 34.8906],
    'רמת השרון': [32.1481, 34.8385],
    'שפרעם': [32.8058, 35.1702],
    'טמרה': [32.8532, 35.1979],
    'נתיבות': [31.4172, 34.5878],
    'מגדל העמק': [32.6732, 35.2415],
    'אופקים': [31.3129, 34.6186],
    'קרית ים': [32.8422, 35.0685],
    'דימונה': [31.0667, 35.0317],
    'יבנה': [31.8745, 34.7405],
    'טירת כרמל': [32.7667, 34.9667],
    'צפת': [32.9646, 35.4960],
    'מעלה אדומים': [31.7770, 35.2995],
    'קרית ביאליק': [32.8375, 35.0833],
    'אור יהודה': [32.0292, 34.8550],
    'קרית אונו': [32.0628, 34.8572],
    'קרית מלאכי': [31.7317, 34.7431],
    'גסר א-זרקא': [32.5367, 34.9125],
    'ג\'סר א-זרקא': [32.5367, 34.9125],
    'ערד': [31.2612, 35.2144],
    'כפר יונה': [32.3168, 34.9351],
    'קרית שמונה': [33.2073, 35.5721],
    'נמל תעופה בן-גוריון': [32.0006, 34.8708],
    'חריש': [32.4633, 35.0450],
    'נשר': [32.7758, 35.0428],
    'מבשרת ציון': [31.7958, 35.1561],
    'גן יבנה': [31.7850, 34.7170],
    'זכרון יעקב': [32.5714, 34.9522],
    'קדימה-צורן': [32.2792, 34.9150],
    'גדרה': [31.8125, 34.7780],
}


# ==========================================
# DATA PIPELINE FUNCTIONS
# ==========================================

def load_crime_df():
    resources = {
        "2025": "e311b6a1-be5a-4a82-8298-f3afbee07b6b",
        "2024": "5fc13c50-b6f3-4712-b831-a75e0f91a17e",
        "2023": "32aacfc9-3524-4fba-a282-3af052380244",
        "2022": "a59f3e9e-a7fe-4375-97d0-76cea68382c1",
        "2021": "3f71fd16-25b8-4cfe-8661-e6199db3eb12",
        "2020": "520597e3-6003-4247-9634-0ae85434b971"
    }
    all_dfs = []
    for year, resource_id in resources.items():
        print(f"loading {year}...")
        url = f"https://data.gov.il/api/3/action/datastore_search"
        limit = 50000
        offset = 0
        rows = []
        while True:
            resp = requests.get(url, params={"resource_id": resource_id, "limit": limit, "offset": offset}, headers=API_HEADERS)
            data = resp.json()
            if 'result' not in data:
                break
            batch = data["result"]["records"]
            rows.extend(batch)
            if len(batch) < limit:
                break 
            offset += limit
        df = pd.DataFrame(rows)
        all_dfs.append(df)

    if not all_dfs:
        raise ValueError("Failed to download any crime data from data.gov.il")
        
    crime_df = pd.concat(all_dfs, ignore_index=True)
    crime_df = crime_df.drop_duplicates()
    return crime_df

def clean_text_columns_regex(df, columns_to_clean):
    df_cleaned = df.copy()
    pattern = r'[^\w\s]|_'
    for col in columns_to_clean:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].apply(
                lambda x: re.sub(pattern, '', str(x)).strip() if pd.notna(x) else x
            )
    return df_cleaned

def strip_whitespace_columns(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df

def impute_missing_names_from_codes(df, code_name_pairs):
    df_out = df.copy()
    for code_col, name_col in code_name_pairs:
        if code_col not in df_out.columns or name_col not in df_out.columns: continue
        missing_before = df_out[name_col].isna().sum()
        if missing_before == 0: continue
        valid_rows = df_out.dropna(subset=[code_col, name_col])
        if valid_rows.empty: continue

        mapping_df = valid_rows[[code_col, name_col]].drop_duplicates(subset=[code_col])
        mapping_dict = dict(zip(mapping_df[code_col], mapping_df[name_col]))
        mask = df_out[name_col].isna() & df_out[code_col].notna()
        df_out.loc[mask, name_col] = df_out.loc[mask, code_col].map(mapping_dict)
    return df_out

def impute_fields_from_station_text(df, station_col, target_cols):
    df_out = df.copy()
    if station_col not in df_out.columns: return df_out
    FORBIDDEN_VALUES = {'', ' ', 'מקום אחר', 'other', 'Other', 'לא ידוע', 'nan', 'None'}

    for target_col in target_cols:
        if target_col not in df_out.columns: continue
        missing_before = df_out[target_col].isna().sum()
        if missing_before == 0: continue

        known_values = df_out[target_col].dropna().unique()
        valid_candidates = [v for v in known_values if str(v).strip() not in FORBIDDEN_VALUES]
        valid_candidates = sorted(valid_candidates, key=lambda x: len(str(x)), reverse=True)
        if not valid_candidates: continue

        mask = df_out[target_col].isna() & df_out[station_col].notna()
        stations_to_check = df_out.loc[mask, station_col].unique()
        mapping = {}
        
        for station in stations_to_check:
            st_str = str(station)
            for candidate in valid_candidates:
                cand_str = str(candidate)
                if cand_str in st_str:
                    mapping[station] = candidate
                    break 
        
        if mapping:
            rows_to_fill = mask & df_out[station_col].isin(mapping.keys())
            df_out.loc[rows_to_fill, target_col] = df_out.loc[rows_to_fill, station_col].map(mapping)
    return df_out

def impute_parent_from_child(df, child_col, parent_col):
    df_out = df.copy()
    if child_col not in df_out.columns or parent_col not in df_out.columns: return df_out
    missing_before = df_out[parent_col].isna().sum()
    if missing_before == 0: return df_out

    valid_relations = df_out.dropna(subset=[child_col, parent_col])[[child_col, parent_col]].drop_duplicates()
    valid_relations = valid_relations[~valid_relations[parent_col].astype(str).isin(['', ' ', 'מקום אחר', 'nan'])]
    ambiguous_children = valid_relations[valid_relations.duplicated(subset=[child_col], keep=False)][child_col].unique()
    
    if len(ambiguous_children) > 0:
        valid_relations = valid_relations[~valid_relations[child_col].isin(ambiguous_children)]
    
    mapping_dict = dict(zip(valid_relations[child_col], valid_relations[parent_col]))
    mask = df_out[parent_col].isna() & df_out[child_col].notna()
    df_out.loc[mask, parent_col] = df_out.loc[mask, child_col].map(mapping_dict)
    return df_out

def get_manual_code_mapping():
    return {
        "נהריה": 9100,
        "קרית גת": 2630,
        "גסר א זרקא": 541,
        "נמל תעופה בן גוריון": 1748,
    }

def inject_manual_codes(df, city_col='Yeshuv', code_col='YeshuvCode'):
    mapping = get_manual_code_mapping()
    if code_col not in df.columns:
        df[code_col] = np.nan
    df = df.copy()
    for city_name, code in mapping.items():
        mask = (df[city_col] == city_name)
        if mask.sum() > 0:
            df.loc[mask, code_col] = code
    return df

def process_and_summarize_crime_data(df):
    impute_pairs = [
        ('municipalKod', 'municipalName'), ('YeshuvKod', 'Yeshuv'),
        ('PoliceDistrictKod', 'PoliceDistrict'), ('PoliceMerhavKod', 'PoliceMerhav'),
        ('StatisticAreaKod', 'StatisticArea'), ('StatisticGroupKod', 'StatisticGroup'),
        ('StatisticTypeKod', 'StatisticType')
    ]
    
    text_cols = [pair[1] for pair in impute_pairs] + ['PoliceStation'] 
    text_cols = [c for c in text_cols if c in df.columns]
    df_processed = clean_text_columns_regex(df, text_cols)
    df_processed = impute_missing_names_from_codes(df_processed, impute_pairs)
    
    df_processed = impute_parent_from_child(df_processed, 'PoliceStation', 'Yeshuv')
    text_mining_targets = ['Yeshuv', 'PoliceMerhav', 'PoliceDistrict', 'municipalName']
    df_processed = impute_fields_from_station_text(df_processed, 'PoliceStation', text_mining_targets)
    
    remaining_steps = [('Yeshuv', 'PoliceMerhav'), ('Yeshuv', 'municipalName'), ('PoliceMerhav', 'PoliceDistrict')]
    for child, parent in remaining_steps:
        df_processed = impute_parent_from_child(df_processed, child, parent)

    df_processed['QuarterYear']= df_processed['Quarter'] +'-'+ df_processed['Year'].astype(str)
    
    group_cols = ['Year', 'Quarter', 'QuarterYear', 'Yeshuv', 'YeshuvKod',
                  'PoliceDistrict', 'PoliceMerhav', 'municipalName', 
                  'StatisticArea', 'StatisticGroup', 'StatisticType']
    valid_group_cols = [c for c in group_cols if c in df_processed.columns]
    
    df_for_agg = df_processed.copy()
    for col in valid_group_cols:
        df_for_agg[col] = df_for_agg[col].fillna('Missing')
    
    summary_df = df_for_agg.groupby(valid_group_cols)['FictiveIDNumber'].count().reset_index()
    summary_df.rename(columns={'FictiveIDNumber': 'EventCount'}, inplace=True)
    return summary_df, df_processed

def fetch_population_data():
    resources = {
        2019: '990ae78e-2dae-4a15-a13b-0b5dcc56056c', 2020: '2d218594-73e3-40de-b36b-23b22f0a2627',
        2021: '95435941-d7e5-46c6-876a-761a74a5928d', 2022: '199b15db-3bcb-470e-ba03-73364737e352',
        2023: 'd47a54ff-87f0-44b3-b33a-f284c0c38e5a'
    }
    base_url = "https://data.gov.il/api/3/action/datastore_search"
    pop_dfs = []
    field_mapping = {
        'Yeshuv_Code': 'סמל יישוב', 'Yeshuv_Name': 'שם יישוב',       
        'Religion_Code': 'דת יישוב', 'Total_Population': 'סך הכל אוכלוסייה',
        'Total_Israelis': 'סך הכל ישראלים', 'Jews_and_Others': 'יהודים ואחרים', 'Arabs': 'ערבים'
    }

    for year, resource_id in resources.items():
        try:
            response = requests.get(base_url, params={'resource_id': resource_id, 'limit': 5000}, headers=API_HEADERS)
            data = response.json()
            if data.get('success'):
                records = data['result']['records']
                df_temp = pd.DataFrame(records)
                found_cols = {}
                name_col = next((c for c in df_temp.columns if 'שם יישוב' in c and 'אנגלית' not in c), None)
                if name_col: found_cols['Yeshuv_Name'] = name_col
                
                for eng_key, heb_search in field_mapping.items():
                    if eng_key == 'Yeshuv_Name': continue 
                    match = next((c for c in df_temp.columns if heb_search in c), None)
                    if match: found_cols[eng_key] = match
                
                if 'Yeshuv_Code' in found_cols or 'Yeshuv_Name' in found_cols:
                    rename_map = {v: k for k, v in found_cols.items()}
                    df_clean = df_temp[list(found_cols.values())].rename(columns=rename_map).copy()
                    df_clean['Year'] = year
                    pop_dfs.append(df_clean)
        except Exception as e:
            print(f"Error fetching population data for {year}: {e}")

    if pop_dfs: return pd.concat(pop_dfs, ignore_index=True)
    return pd.DataFrame()

def preprocess_population(df):
    if df.empty: return df
    df['Yeshuv_Code'] = pd.to_numeric(df['Yeshuv_Code'], errors='coerce')
    df = strip_whitespace_columns(df, ['Yeshuv_Name'])
    
    numeric_cols = ['Total_Population', 'Total_Israelis', 'Jews_and_Others', 'Arabs', 'Religion_Code']
    for col in numeric_cols:
        if col in df.columns:
            if df[col].dtype == object:
                 df[col] = df[col].astype(str).str.replace(',', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def extrapolate_quarters(pop_df, years_of_interest):
    expanded_data = []
    valid_pop = pop_df.dropna(subset=['Yeshuv_Code']).copy()
    
    for yeshuv_code, group in valid_pop.groupby('Yeshuv_Code'):
        group = group.sort_values('Year')
        yeshuv_name = group['Yeshuv_Name'].iloc[0]
        for year in years_of_interest:
            current_row = group[group['Year'] == year]
            if current_row.empty: continue
            
            curr_pop = current_row['Total_Population'].values[0]
            if pd.isna(curr_pop): curr_pop = 0
            
            rel_code = current_row['Religion_Code'].values[0] if 'Religion_Code' in current_row else np.nan
            israelis = current_row['Total_Israelis'].values[0] if 'Total_Israelis' in current_row else np.nan
            jews = current_row['Jews_and_Others'].values[0] if 'Jews_and_Others' in current_row else np.nan
            arabs = current_row['Arabs'].values[0] if 'Arabs' in current_row else np.nan

            prev_row = group[group['Year'] == year - 1]
            growth_calculated = False
            if not prev_row.empty:
                prev_pop = prev_row['Total_Population'].values[0]
                if pd.notna(prev_pop):
                    diff = curr_pop - prev_pop
                    q_growth = diff / 4
                    quarters = [prev_pop + q_growth, prev_pop + (q_growth*2), prev_pop + (q_growth*3), curr_pop]
                    growth_calculated = True
                    
            if not growth_calculated: quarters = [curr_pop] * 4
            
            for q_idx, q_name in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
                val = quarters[q_idx]
                expanded_data.append({
                    'Yeshuv_Code': yeshuv_code, 'Yeshuv_Name': yeshuv_name, 'Year': year, 'Quarter': q_name,
                    'Total_Population': int(val) if pd.notna(val) else 0, 
                    'Religion_Code': rel_code, 'Total_Israelis': israelis,
                    'Jews_and_Others': jews, 'Arabs': arabs
                })
    return pd.DataFrame(expanded_data)

def join_crime_population(crime_df_agg, pop_quarterly_df):
    crime_df_clean = crime_df_agg.dropna(subset=['Yeshuv', 'Year']).copy()
    crime_df_clean['Join_Key_Code'] = pd.to_numeric(crime_df_clean['YeshuvKod'], errors='coerce')
    crime_df_clean['Join_Key_Name'] = crime_df_clean['Yeshuv'].astype(str).str.strip()
    
    merge_cols = ['Yeshuv_Code', 'Year', 'Quarter', 'Total_Population', 'Religion_Code', 
                  'Total_Israelis', 'Jews_and_Others', 'Arabs']
    available_cols = [c for c in merge_cols if c in pop_quarterly_df.columns]
    
    merged_df = pd.merge(
        crime_df_clean, pop_quarterly_df[available_cols],
        left_on=['Join_Key_Code', 'Year', 'Quarter'], right_on=['Yeshuv_Code', 'Year', 'Quarter'], how='left'
    )
    
    name_merge_cols = ['Yeshuv_Name', 'Year', 'Quarter'] + [c for c in available_cols if c not in ['Yeshuv_Code', 'Year', 'Quarter']]
    merged_with_name = pd.merge(
        merged_df, pop_quarterly_df[name_merge_cols],
        left_on=['Join_Key_Name', 'Year', 'Quarter'], right_on=['Yeshuv_Name', 'Year', 'Quarter'],
        how='left', suffixes=('', '_NameFallback')
    )
    
    target_fields = ['Total_Population', 'Religion_Code', 'Total_Israelis', 'Jews_and_Others', 'Arabs']
    for col in target_fields:
        fallback_col = f'{col}_NameFallback'
        if fallback_col in merged_with_name.columns:
             merged_with_name[col] = merged_with_name[col].fillna(merged_with_name[fallback_col])
    
    cols_to_drop = [c for c in merged_with_name.columns if 'Join_Key' in c or '_NameFallback' in c]
    merged_final = merged_with_name.drop(columns=cols_to_drop, errors='ignore')

    if 'Yeshuv' in merged_final.columns and 'Yeshuv_Name' in merged_final.columns:
        merged_final['Yeshuv'] = merged_final['Yeshuv'].fillna(merged_final['Yeshuv_Name'])
        merged_final.drop(columns=['Yeshuv_Name'], inplace=True)
    if 'YeshuvKod' in merged_final.columns and 'Yeshuv_Code' in merged_final.columns:
        merged_final['YeshuvKod'] = merged_final['YeshuvKod'].fillna(merged_final['Yeshuv_Code'])
        merged_final.drop(columns=['Yeshuv_Code'], inplace=True)

    return merged_final

def get_chained_quarterly_cpi(start_year, end_year):
    cpi_id = 120010
    url = "https://api.cbs.gov.il/index/data/price"
    params = {"id": cpi_id, "startPeriod": f"01-{start_year}", "endPeriod": f"12-{end_year}", "format": "json", "download": "false"}
    try:
        response = requests.get(url, params=params, headers=API_HEADERS)
        data = response.json()
        if 'month' not in data or not data['month']: return None

        observations = data['month'][0].get('date', [])
        records = []
        for obs in observations:
            pct_change = obs.get('percent')
            if pct_change is not None:
                records.append({'date': f"{obs['year']}-{obs['month']:02d}-01", 'monthly_percent_change': float(pct_change)})
        
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df['chained_index'] = 100.0
        
        for i in range(1, len(df)):
            df.loc[i, 'chained_index'] = df.loc[i-1, 'chained_index'] * (1 + df.loc[i, 'monthly_percent_change'] / 100)

        quarterly_df = df.set_index('date').resample('Q').agg({
            'chained_index': 'mean', 'monthly_percent_change': 'sum'
        }).reset_index()
        
        quarterly_df['quarter_name'] = quarterly_df['date'].dt.to_period('Q')
        quarterly_df = quarterly_df.rename(columns={'chained_index': 'avg_chained_index_points', 'monthly_percent_change': 'total_quarterly_inflation_pct'})
        return quarterly_df[['quarter_name', 'avg_chained_index_points', 'total_quarterly_inflation_pct']]
    except Exception as e:
        print(f"Error fetching CPI: {e}")
        return None

def run_data_pipeline(final_crime_path, cpi_path):
    print("Initiating Pipeline...")
    
    # 1. Fetch & Process Crime Data
    crime_df = load_crime_df()
    crime_df_agg, _ = process_and_summarize_crime_data(crime_df)

    # 2. Fetch & Process Pop Data
    pop_raw = fetch_population_data()
    pop_clean = preprocess_population(pop_raw)
    pop_quarterly = extrapolate_quarters(pop_clean, years_of_interest=[2020, 2021, 2022, 2023])

    # 3. Merge & Save as GZIP compressed CSV
    if not pop_quarterly.empty and crime_df_agg is not None:
        final_df = join_crime_population(crime_df_agg, pop_quarterly)
        final_df.to_csv(final_crime_path, index=False, encoding='utf-8-sig', compression='gzip')

    # 4. Fetch & Save CPI
    cpi_df = get_chained_quarterly_cpi(2020, 2025)
    if cpi_df is not None:
        cpi_df.to_csv(cpi_path, index=False, encoding='utf-8-sig')
    print("Pipeline Complete!")


# ==========================================
# DASHBOARD FUNCTIONS
# ==========================================

def determine_majority_religion(df):
    """
    Attempts to determine the majority religion per row based on exact column names provided.
    Includes mapping to English for uniform UI presentation.
    """
    potential_cols = ['Jews_and_Others', 'Arabs', 'יהודים ואחרים', 'ערבים']
    existing_cols = [col for col in potential_cols if col in df.columns]
    
    if len(existing_cols) >= 2:
        temp_df = df[existing_cols].replace({',': ''}, regex=True).apply(pd.to_numeric, errors='coerce').fillna(0)
        majority = temp_df.idxmax(axis=1)
        
        name_mapping = {
            'יהודים ואחרים': 'Jews & Others',
            'Jews_and_Others': 'Jews & Others',
            'ערבים': 'Arabs',
            'Arabs': 'Arabs'
        }
        df['Majority_Religion'] = majority.map(name_mapping).fillna('Data Not Available')
    else:
        df['Majority_Religion'] = 'Data Not Available'
        
    return df

@st.cache_data(show_spinner=False)
def load_cloud_ready_data(crime_path, cpi_path):
    """
    Cloud-optimized data loader. 
    Checks if files exist. If yes, loads them. 
    If no, safely executes the pipeline, caches the result in memory, and saves locally.
    """
    if not os.path.exists(crime_path) or not os.path.exists(cpi_path):
        run_data_pipeline(crime_path, cpi_path)
        
    # Pandas natively handles '.gz' compression automatically!
    df = pd.read_csv(crime_path, low_memory=False, compression='gzip')
    try:
        cpi_df = pd.read_csv(cpi_path)
    except FileNotFoundError:
        cpi_df = None
        
    return df, cpi_df

def visualize_crime_rates_streamlit(merged_df, cpi_df=None):
    # Removed st.set_page_config from here!
    st.title("🛡️ Israel Crime Analysis")

    # --- Create a distinct and large pastel color palette so colors never repeat ---
    DISTINCT_COLORS = px.colors.qualitative.Pastel + px.colors.qualitative.Set3 + px.colors.qualitative.Pastel1 + px.colors.qualitative.Pastel2

    # --- 1. Data Preprocessing ---
    # Strip column names of hidden whitespaces
    merged_df.columns = merged_df.columns.str.strip()
    
    valid_years = [2020, 2021, 2022, 2023]
    df_clean = merged_df[merged_df['Year'].isin(valid_years)].copy()
    
    # Filter out "שגיאת הזנה" (Input Error)
    df_clean = df_clean[df_clean['StatisticGroup'] != 'שגיאת הזנה']
    
    df_clean['EventCount'] = pd.to_numeric(df_clean['EventCount'], errors='coerce').fillna(0)
    df_clean['Total_Population'] = pd.to_numeric(df_clean['Total_Population'], errors='coerce')
    df_clean = determine_majority_religion(df_clean)

    # --- 2. SIDEBAR FILTERS (All Dropdowns) ---
    st.sidebar.header("Global Data Filters")
    st.sidebar.markdown("Changes here affect all visualizations.")

    year_options = ["Average", 2020, 2021, 2022, 2023]
    selected_year = st.sidebar.selectbox("Select Year", options=year_options, index=0)

    quarter_options = ["All Quarters"]
    if 'Quarter' in df_clean.columns:
        quarter_options += sorted(df_clean['Quarter'].dropna().unique().tolist())
    selected_quarter = st.sidebar.selectbox("Select Quarter", options=quarter_options, index=0)

    all_groups = ["All Groups"] + sorted(df_clean['StatisticGroup'].dropna().unique().tolist())
    selected_group = st.sidebar.selectbox("Select Crime Group (Type)", options=all_groups, index=0)
    
    if selected_group == "All Groups":
        available_subtypes = ["All Sub-Types"] + sorted(df_clean['StatisticType'].dropna().unique().tolist())
    else:
        available_subtypes = ["All Sub-Types"] + sorted(df_clean[df_clean['StatisticGroup'] == selected_group]['StatisticType'].dropna().unique().tolist())
    selected_subtype = st.sidebar.selectbox("Select Crime Sub-Type", options=available_subtypes, index=0)

    all_religions = ["All Populations"] + sorted(df_clean['Majority_Religion'].dropna().unique().tolist())
    selected_religion = st.sidebar.selectbox("Select Population Majority", options=all_religions, index=0)

    # --- Apply Filters ---
    df_filtered = df_clean.copy()
    if selected_quarter != "All Quarters":
        df_filtered = df_filtered[df_filtered['Quarter'] == selected_quarter]
    if selected_group != "All Groups":
        df_filtered = df_filtered[df_filtered['StatisticGroup'] == selected_group]
    if selected_subtype != "All Sub-Types":
        df_filtered = df_filtered[df_filtered['StatisticType'] == selected_subtype]
    if selected_religion != "All Populations":
        df_filtered = df_filtered[df_filtered['Majority_Religion'] == selected_religion]

    # --- 3. Base Aggregation Logic ---
    agg_yearly = df_filtered.groupby(['Year', 'Yeshuv', 'Majority_Religion']).agg({
        'EventCount': 'sum',
        'Total_Population': 'max'
    }).reset_index()

    agg_yearly['Crime_Rate'] = (agg_yearly['EventCount'] / agg_yearly['Total_Population']) * 1000
    agg_yearly['Crime_Percentage'] = (agg_yearly['EventCount'] / agg_yearly['Total_Population']) * 100
    agg_yearly = agg_yearly[agg_yearly['Total_Population'] > 0].dropna(subset=['Crime_Rate'])

    agg_avg = agg_yearly.groupby(['Yeshuv', 'Majority_Religion']).agg({
        'EventCount': 'mean',
        'Total_Population': 'mean',
        'Crime_Rate': 'mean',
        'Crime_Percentage': 'mean'
    }).reset_index()
    agg_avg['Year'] = 'Average'

    if selected_year == "Average":
        current_view_df = agg_avg.copy()
        period_label = f"Average (2020-2023)"
    else:
        current_view_df = agg_yearly[agg_yearly['Year'] == selected_year].copy()
        period_label = str(selected_year)

    if selected_quarter != "All Quarters":
        period_label += f" | {selected_quarter}"

    # --- 4. METRICS SECTION ---
    total_crimes = int(current_view_df['EventCount'].sum()) if not current_view_df.empty else 0
    avg_rate = current_view_df['Crime_Rate'].mean() if not current_view_df.empty else 0.0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"Total Crimes ({selected_year})", f"{total_crimes:,}")
    col2.metric("Avg National Crime Rate*", f"{avg_rate:.2f}" if not np.isnan(avg_rate) else "0.00")
    col3.metric("Selected Period", period_label)
    col4.metric("Selected Population", selected_religion)
    
    st.caption("* Crime Rate = crime incidents per 1,000 residents")

    st.markdown("---")

    # ==========================================
    # VISUALIZATIONS
    # ==========================================

    # --- 1. TREEMAP: NATIONAL CRIME COMPOSITION ---
    st.header("1. 📊 National Crime Distribution - Most crimes are 'Against Public Order', threats specifically with 14%")
    
    if selected_year == "Average":
        treemap_agg = df_filtered.groupby(['Year', 'StatisticGroup', 'StatisticType'])['EventCount'].sum().reset_index()
        treemap_agg = treemap_agg.groupby(['StatisticGroup', 'StatisticType'])['EventCount'].mean().reset_index()
    else:
        tree_df = df_filtered[df_filtered['Year'] == selected_year]
        treemap_agg = tree_df.groupby(['StatisticGroup', 'StatisticType'])['EventCount'].sum().reset_index()

    if not treemap_agg.empty and treemap_agg['EventCount'].sum() > 0:
        total_period_events = treemap_agg['EventCount'].sum()
        treemap_agg['Share'] = (treemap_agg['EventCount'] / total_period_events) * 100
        
        fig_tree = px.treemap(
            treemap_agg, path=[px.Constant("All Crimes"), 'StatisticGroup', 'StatisticType'],
            values='EventCount', color='StatisticGroup',
            title=f"Crime Breakdown - {period_label}", color_discrete_sequence=DISTINCT_COLORS
        )
        
        fig_tree.update_traces(
            textinfo="label+value+percent root", 
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Count: %{value:,.0f}<br>"
                "Share of Total Crime: %{percentRoot:.1%}<br>"
                "Share of Category: %{percentParent:.1%}<extra></extra>"
            )
        )
        fig_tree.update_layout(margin=dict(t=50, l=25, r=25, b=25), height=600)
        st.plotly_chart(fig_tree, use_container_width=True)
    else:
        st.info("No data available for the selected filter.")

    st.markdown("---")

    # --- 2. OVERALL CRIME RATE TREND OVER TIME ---
    st.header("2. 📉 Overall Crime Rate Trend Over Time - 2023 had even lower crime rate than COVID lockdown in 2021")
    st.markdown("Examines whether the total volume of crime (normalized per 1,000 residents) is changing. Note the volatility of the data: it is not a clean trend, but more like strains by each year, as this is raw data without processing.")
    if 'Quarter' in df_filtered.columns:
        trend_base = df_filtered.groupby(['Year', 'Quarter', 'Yeshuv']).agg({
            'EventCount': 'sum',
            'Total_Population': 'max'
        }).reset_index()
        
        trend_base['YearQuarter'] = trend_base['Year'].astype(str) + "-" + trend_base['Quarter']
        nat_trend = trend_base.groupby('YearQuarter').agg({
            'EventCount': 'sum',
            'Total_Population': 'sum'
        }).reset_index()
        
        nat_trend = nat_trend[nat_trend['Total_Population'] > 0]
        nat_trend['National_Crime_Rate'] = (nat_trend['EventCount'] / nat_trend['Total_Population']) * 1000
        nat_trend['TextLabel'] = nat_trend['National_Crime_Rate'].apply(lambda x: f"{x:.2f}")
        
        fig_nat_trend = px.line(
            nat_trend, x='YearQuarter', y='National_Crime_Rate', markers=True, text='TextLabel',
            title="Normalized Crime Rate Over Time (per 1,000 Residents)",
            labels={'YearQuarter': 'Quarter', 'National_Crime_Rate': 'Crime Rate (per 1k)'}
        )
        fig_nat_trend.update_traces(textposition="top center", line_color='#1f77b4', marker=dict(size=8))
        fig_nat_trend.update_layout(yaxis_title="Crime Rate (per 1,000 Residents)")
        st.plotly_chart(fig_nat_trend, use_container_width=True)

    st.markdown("---")

    # --- 3. CHANGE IN DISTRIBUTION OVER TIME (Absolute Volume + Percentage inside) ---
    st.header("3. 📈 Crime Volume & Distribution Over Time - the share of crimes against property is increasing with time")
    st.markdown("Displays the absolute number of crimes per quarter, with internal segments representing the percentage of each crime type. There's no internal annual trend of quarters/seasons.")
    if 'Quarter' in df_filtered.columns:
        q1_df = df_filtered.groupby(['Year', 'Quarter', 'StatisticGroup'])['EventCount'].sum().reset_index()
        q1_df['YearQuarter'] = q1_df['Year'].astype(str) + "-" + q1_df['Quarter']
        q1_total = q1_df.groupby('YearQuarter')['EventCount'].transform('sum')
        q1_df['Percent'] = (q1_df['EventCount'] / q1_total) * 100
        
        # Determine global sort order (Largest volume at the bottom of the stack)
        cat_order = q1_df.groupby('StatisticGroup')['EventCount'].sum().sort_values(ascending=False).index.tolist()
        
        # Add a text label, hide text for tiny segments to avoid cluttering the visual
        q1_df['TextLabel'] = q1_df['Percent'].apply(lambda x: f"{x:.0f}%" if x >= 3 else "")

        fig_q1 = px.bar(
            q1_df, x='YearQuarter', y='EventCount', color='StatisticGroup',
            title="Absolute Crime Volume with Relative Distribution",
            labels={'YearQuarter': 'Quarter', 'EventCount': 'Total Crimes (Absolute)', 'StatisticGroup': 'Crime Type'},
            text='TextLabel',
            category_orders={'StatisticGroup': cat_order}, # Ensures largest segments are at the bottom
            color_discrete_sequence=DISTINCT_COLORS
        )
        fig_q1.update_traces(textposition='inside', textfont_size=12)
        fig_q1.update_layout(barmode='stack')
        st.plotly_chart(fig_q1, use_container_width=True)

    st.markdown("---")

    # --- 4. CPI VS PROPERTY CRIME ---
    st.header("4. 💰 Correlation Observed Between CPI and Share of Property Crimes")
    if cpi_df is not None and not cpi_df.empty:
        cpi_df.columns = cpi_df.columns.str.strip()
        
        # Group crime by Quarter and StatisticGroup
        q4_df = df_clean.groupby(['Year', 'Quarter', 'StatisticGroup'])['EventCount'].sum().reset_index()
        q4_df['YearQuarter'] = q4_df['Year'].astype(str) + q4_df['Quarter']
        
        q4_pivot = q4_df.pivot(index='YearQuarter', columns='StatisticGroup', values='EventCount').fillna(0)
        q4_pct = q4_pivot.div(q4_pivot.sum(axis=1), axis=0) * 100
        
        # Look for the 'Property' / 'רכוש' column
        prop_cols = [c for c in q4_pct.columns if 'רכוש' in str(c) or 'Property' in str(c)]
        
        if prop_cols and 'quarter_name' in cpi_df.columns and 'avg_chained_index_points' in cpi_df.columns:
            col_name = prop_cols[0]
            cpi_df['quarter_name'] = cpi_df['quarter_name'].astype(str).str.strip()
            
            merged_cpi = q4_pct[[col_name]].merge(cpi_df, left_index=True, right_on='quarter_name')

            fig_q4 = go.Figure()

            try:
                import statsmodels.api as sm
                has_sm = True
            except ImportError:
                has_sm = False
                st.warning("Trendline and Confidence Intervals are hidden because 'statsmodels' is not installed. Run `pip install statsmodels` in your terminal.")

            cat_data = merged_cpi.dropna(subset=['avg_chained_index_points', col_name])
            cat_data = cat_data.sort_values('avg_chained_index_points')
            
            x_vals = cat_data['avg_chained_index_points']
            y_vals = cat_data[col_name]
            
            # 1. Add Scatter Points
            fig_q4.add_trace(go.Scatter(
                x=x_vals, y=y_vals, mode='markers', name='Property Crimes',
                marker=dict(color='royalblue', size=10)
            ))
            
            # 2. Add Regression Line & Confidence Interval if statsmodels is available
            if has_sm:
                X_sm = sm.add_constant(x_vals)
                model = sm.OLS(y_vals, X_sm).fit()
                predictions = model.get_prediction(X_sm)
                pred_df = predictions.summary_frame(alpha=0.05) # 95% Confidence Interval
                
                # Trendline
                fig_q4.add_trace(go.Scatter(
                    x=x_vals, y=pred_df['mean'], mode='lines', name='Trend',
                    line=dict(color='royalblue', width=2)
                ))
                
                # Confidence Band (Polygon connecting upper and lower bounds)
                x_list = x_vals.tolist()
                x_band = x_list + x_list[::-1]
                
                y_upper = pred_df['mean_ci_upper'].tolist()
                y_lower = pred_df['mean_ci_lower'].tolist()
                y_band = y_upper + y_lower[::-1]
                
                fig_q4.add_trace(go.Scatter(
                    x=x_band, y=y_band, fill='toself', fillcolor='rgba(65, 105, 225, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'), hoverinfo='skip', showlegend=False,
                    name='95% CI'
                ))

            fig_q4.update_layout(
                title="Consumer Price Index (CPI) vs. Percentage of Property Crimes",
                xaxis_title="Consumer Price Index (CPI Points)",
                yaxis_title="Percentage of Property Crimes (%)",
                hovermode="x unified"
            )

            st.plotly_chart(fig_q4, use_container_width=True)
        else:
            st.info("Missing specific columns: Could not map 'Property/רכוש' crime, 'quarter_name', or 'avg_chained_index_points' in the provided files.")
    else:
        st.warning("⚠️ Quarterly CPI data (quarterly_cpi_chained.csv) was not found or is empty.")

    st.markdown("---")

    # --- 5. INTERACTIVE MAP SECTION ---
    st.header("5. 🗺️ Crime rates are higher in suburbs compared to the center")
    
    map_data = current_view_df.copy()
    def get_lat_lon(city_name):
        return CITY_COORDINATES.get(city_name, [None, None])

    map_data['coords'] = map_data['Yeshuv'].apply(get_lat_lon)
    map_data[['lat', 'lon']] = pd.DataFrame(map_data['coords'].tolist(), index=map_data.index)
    map_plot_df = map_data.dropna(subset=['lat', 'lon']).copy()
    
    if map_plot_df.empty:
        st.warning("No geographic data matched for the selected subset.")
    else:
        map_plot_df['fmt_Crime_Rate'] = map_plot_df['Crime_Rate'].round(0).astype(int).astype(str)
        map_plot_df['fmt_Crime_Percentage'] = map_plot_df['Crime_Percentage'].round(0).astype(int).astype(str) + '%'
        map_plot_df['fmt_EventCount'] = map_plot_df['EventCount'].round(0).astype(int).astype(str)
        map_plot_df['fmt_Total_Population'] = map_plot_df['Total_Population'].round(0).astype(int).astype(str)

        def get_color_by_rate(rate):
            if rate < 10: return [224, 243, 252, 200]
            elif rate < 20: return [158, 202, 225, 200]
            elif rate < 30: return [66, 146, 198, 200]
            elif rate < 40: return [8, 81, 156, 200]
            else: return [8, 48, 107, 220]

        map_plot_df['color'] = map_plot_df['Crime_Rate'].apply(get_color_by_rate)
        
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 10px; flex-wrap: wrap;">
                <span style="font-weight:bold;">Crime Rate (per 1k):</span>
                <span style="color:#e0f3fc;">■ < 10 (Very Low)</span>
                <span style="color:#9ecae1;">■ 10-20 (Low)</span>
                <span style="color:#4292c6;">■ 20-30 (Medium)</span>
                <span style="color:#08519c;">■ 30-40 (High)</span>
                <span style="color:#08306b;">■ > 40 (Severe)</span>
            </div>
            """, unsafe_allow_html=True
        )

        layer = pdk.Layer(
            "ScatterplotLayer", map_plot_df, get_position='[lon, lat]',
            get_color='color', get_radius=1500, radius_min_pixels=5,
            radius_max_pixels=40, pickable=True, auto_highlight=True,
        )

        tooltip = {
            "html": """
                <div style="font-family: sans-serif; padding: 4px; color: white;">
                    <b>{Yeshuv}</b><br/>
                    Majority Religion: <b>{Majority_Religion}</b><br/>
                    Crime Rate: <b>{fmt_Crime_Rate}</b><br/>
                    Crime Percentage: <b>{fmt_Crime_Percentage}</b><br/>
                    Events: {fmt_EventCount}<br/>
                    Population: {fmt_Total_Population}
                </div>
            """,
            "style": {"backgroundColor": "#111", "borderRadius": "5px"}
        }

        view_state = pdk.ViewState(latitude=31.5, longitude=34.8, zoom=6.5, pitch=0)

        st.pydeck_chart(pdk.Deck(
            layers=[layer], initial_view_state=view_state, tooltip=tooltip,
            map_style='https://basemaps.cartocdn.com/gl/positron-gl-style/style.json'
        ))

    st.markdown("---")

    # --- 6. TOP 10 CITIES ---
    st.header("6. 📍 Top 10 Cities by Crime Rate - 4 out of 10 cities are Arab majority")
    min_pop = st.slider("Minimum City Population Filter", 1000, 50000, 5000, step=1000)
    
    filtered_cities = current_view_df[current_view_df['Total_Population'] >= min_pop]
    top_20 = filtered_cities.sort_values('Crime_Rate', ascending=False).head(10)
    
    if not top_20.empty:
        fig_bar = px.bar(
            top_20, x='Crime_Rate', y='Yeshuv', orientation='h',
            title=f"Top 10 Cities by Crime Rate ({period_label}) [Pop > {min_pop}]",
            color='Crime_Rate', color_continuous_scale='Blues',
            labels={'Crime_Rate': 'Crime Rate (per 1k)', 'Yeshuv': 'City'},
            hover_data=['Majority_Religion', 'Total_Population']
        )
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, height=500)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No cities meet the population criteria with the current filters.")

    st.markdown("---")

    # --- 7. SHARPEST CHANGE IN RATE ---
    st.header("7. 📊 Cities with the Sharpest Change in Crime Rate (2020-2023)")
    df_change = df_clean[df_clean['Year'].isin([2020, 2023])].copy()
    change_agg = df_change.groupby(['Yeshuv', 'Year']).agg({'EventCount':'sum', 'Total_Population':'max'}).reset_index()
    change_agg = change_agg[change_agg['Total_Population'] > 0]
    change_agg['Rate'] = (change_agg['EventCount'] / change_agg['Total_Population']) * 1000
    
    pivot_change = change_agg.pivot(index='Yeshuv', columns='Year', values='Rate').dropna()
    if 2020 in pivot_change.columns and 2023 in pivot_change.columns:
        pivot_change['RateChange'] = pivot_change[2023] - pivot_change[2020]
        top_inc = pivot_change.nlargest(5, 'RateChange').reset_index()
        top_dec = pivot_change.nsmallest(5, 'RateChange').reset_index()
        combined_change = pd.concat([top_inc, top_dec]).sort_values('RateChange')
        
        combined_change['Color'] = combined_change['RateChange'].apply(lambda x: 'Decrease (Green)' if x < 0 else 'Increase (Red)')
        
        fig_q6 = px.bar(
            combined_change, x='RateChange', y='Yeshuv', orientation='h',
            color='Color', color_discrete_map={'Decrease (Green)': 'green', 'Increase (Red)': 'red'},
            title="Change in Crime Rate per 1,000 Residents (2020-2023)",
            labels={'RateChange': 'Change in Crime Rate (Points)', 'Yeshuv': 'City', 'Color': 'Trend'},
            text=combined_change['RateChange'].apply(lambda x: f"{x:+.0f}")
        )
        st.plotly_chart(fig_q6, use_container_width=True)
    else:
        st.info("Data for 2020 or 2023 is missing to calculate the change.")

    st.markdown("---")

    # --- 8. CRIME COMPOSITION BY DEMOGRAPHIC (BOXPLOT) ---
    st.header("8. 📊 Correlation Observed Between Locality Demographic and Dominant Crime Profile")
    st.markdown("Compares the distribution of crime types across individual cities, grouped by majority demographic (Boxplot shows median, quartiles, and outliers per city).")

    df_demo = df_filtered[df_filtered['Majority_Religion'].isin(['Jews & Others', 'Arabs'])].copy()
    if not df_demo.empty:
        # Calculate crimes per city per category
        city_crime_type = df_demo.groupby(['Yeshuv', 'Majority_Religion', 'StatisticGroup'])['EventCount'].sum().reset_index()
        
        # Calculate total crimes per city
        city_total = df_demo.groupby('Yeshuv')['EventCount'].sum().reset_index().rename(columns={'EventCount': 'CityTotal'})
        
        # Merge and calculate percentage
        city_comp = city_crime_type.merge(city_total, on='Yeshuv')
        city_comp = city_comp[city_comp['CityTotal'] > 0]
        city_comp['Percent'] = (city_comp['EventCount'] / city_comp['CityTotal']) * 100
        
        # Get count of unique cities for context
        city_counts = city_comp[['Yeshuv', 'Majority_Religion']].drop_duplicates()['Majority_Religion'].value_counts()
        jews_count = city_counts.get('Jews & Others', 0)
        arabs_count = city_counts.get('Arabs', 0)
        
        st.info(f"💡 **Analysis Scope:** Unique localities included in this comparison: **{jews_count}** Jewish & Others localities, and **{arabs_count}** Arab localities.")

        # Sort categories by overall median for better readability
        cat_order = city_comp.groupby('StatisticGroup')['Percent'].median().sort_values(ascending=False).index.tolist()

        fig_demo = px.box(
            city_comp, x='StatisticGroup', y='Percent', color='Majority_Religion',
            title="Demographic Variance in Crime Type Distribution: Jewish vs. Arab Localities",
            labels={
                'StatisticGroup': 'Crime Type', 
                'Percent': '% of City\'s Total Crimes', 
                'Majority_Religion': 'Majority Population'
            },
            category_orders={'StatisticGroup': cat_order},
            color_discrete_map={
                'Jews & Others': px.colors.qualitative.Pastel[1], # פסטל תכלת
                'Arabs': px.colors.qualitative.Pastel[4]          # פסטל כתום
            }
        )
        
        fig_demo.update_layout(xaxis_tickangle=-45, height=600)
        st.plotly_chart(fig_demo, use_container_width=True)
    else:
        st.info("Not enough demographic data available to display this comparison.")

    st.markdown("---")

    # --- 9. COVID AND WAR IMPACT ---
    st.header("9. 🦠⚔️ Crime Distribution Across National Periods - Property crime shares were lower during emergencies compared to routine")
    def get_period(row):
        y = row.get('Year')
        q = row.get('Quarter', '')
        if y in [2020, 2021]: return "COVID-19 Period"
        if (y == 2023 and q == 'Q4') or y == 2024: return "Iron Swords War"
        return "Routine"
    
    df_q2 = df_filtered.copy()
    if not df_q2.empty:
        df_q2['Period'] = df_q2.apply(get_period, axis=1)
        q2_df = df_q2.groupby(['Period', 'StatisticGroup'])['EventCount'].sum().reset_index()
        q2_total = q2_df.groupby('Period')['EventCount'].transform('sum')
        q2_df['Percent'] = (q2_df['EventCount'] / q2_total) * 100
        
        # Sort values to ensure lines draw correctly left-to-right based on period order
        period_order = ['COVID-19 Period', 'Routine', 'Iron Swords War']
        q2_df['Period'] = pd.Categorical(q2_df['Period'], categories=period_order, ordered=True)
        q2_df = q2_df.sort_values(['StatisticGroup', 'Period'])

        # Filter out labels that are too small to prevent clutter, or format them cleanly
        q2_df['TextLabel'] = q2_df['Percent'].apply(lambda x: f"{x:.1f}%" if x >= 1 else "")

        # Use a line chart (Slope Chart) instead of bar chart
        fig_q2 = px.line(
            q2_df, x='Period', y='Percent', color='StatisticGroup', markers=True,
            title="Crime Distribution Across Different National Periods (Slope Chart)",
            labels={'Period': 'Time Period', 'Percent': '% of Total Crime', 'StatisticGroup': 'Crime Type'},
            text='TextLabel', 
            category_orders={
                'Period': period_order
            },
            color_discrete_sequence=DISTINCT_COLORS
        )
        
        fig_q2.update_traces(
            textposition='top center', 
            textfont_size=11, 
            line=dict(width=3), 
            marker=dict(size=8)
        )
        fig_q2.update_layout(
            xaxis_title="Time Period", 
            yaxis_title="Percentage of Total Crime (%)",
            height=800  # Stretched the chart to spread out the lines and avoid overlap
        )
        st.plotly_chart(fig_q2, use_container_width=True)

    st.markdown("---")

    # --- 10. SOCIO-ECONOMIC CORRELATION ---
    cluster_cols = [c for c in df_clean.columns if 'cluster' in c.lower() or 'socio' in c.lower() or 'אשכול' in c]
    if cluster_cols:
        st.header("10. 🏙️ Correlation Between Socio-Economic Cluster and Crime Rate")
        cluster_col = cluster_cols[0]
        
        df_q5 = agg_avg.copy() if selected_year == "Average" else agg_yearly[agg_yearly['Year'] == selected_year].copy()
        cluster_map = df_clean[['Yeshuv', cluster_col]].dropna().drop_duplicates(subset=['Yeshuv'])
        df_q5 = df_q5.merge(cluster_map, on='Yeshuv', how='inner')
        
        if not df_q5.empty:
            fig_q5 = px.scatter(
                df_q5, x=cluster_col, y='Crime_Rate', size='EventCount', hover_name='Yeshuv',
                title="Socio-Economic Cluster vs. Crime Rate",
                labels={cluster_col: 'Socio-Economic Cluster (1=Low)', 'Crime_Rate': 'Cases per 1,000 Residents', 'EventCount': 'Total Crimes'}
            )
            st.plotly_chart(fig_q5, use_container_width=True)
            st.markdown("---")

    # --- DATA TABLE ---
    with st.expander("View Raw Aggregated Data"):
        st.dataframe(current_view_df.sort_values(['Crime_Rate'], ascending=[False]))

# --- Main App Execution ---
if __name__ == "__main__":
    # Updated file extension to .csv.gz to handle compression
    final_crime_path = "merged_crime_population_final.csv.gz"
    cpi_path = "quarterly_cpi_chained.csv"

    try:
        if not os.path.exists(final_crime_path) or not os.path.exists(cpi_path):
            st.info("⚠️ Generated CSVs not found locally. Initiating automated Data Pipeline... (This may take memory & time)")
            st.warning("💡 Hint: If the app crashes shortly after this, the cloud server likely ran out of RAM. Best fix: Run the script on your PC and upload the generated CSV files directly to GitHub.")
        
        with st.spinner("Loading and processing dashboard data..."):
            df, cpi_df = load_cloud_ready_data(final_crime_path, cpi_path)
            
        if df is not None and not df.empty:
            visualize_crime_rates_streamlit(df, cpi_df)
        else:
            st.error("Dashboard could not load because the dataset is empty or failed to process.")

    except Exception as e:
        st.error(f"An unexpected error occurred while loading the app: {e}")
        st.info("Please make sure all dependencies in your `requirements.txt` are installed.")
