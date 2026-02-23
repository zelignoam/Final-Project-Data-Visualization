import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import numpy as np
import sys
import os

# --- Static Data: Coordinates for Major Israeli Cities ---
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

def determine_majority_religion(df):
    """
    Attempts to determine the majority religion per row based on exact column names provided.
    Includes mapping to English for uniform UI presentation.
    """
    potential_cols = ['Jews_and_Others', 'Arabs', 'יהודים ואחרים', 'ערבים']
    existing_cols = [col for col in potential_cols if col in df.columns]
    
    if len(existing_cols) >= 2:
        # Strip commas from numbers (e.g., "45,000" -> "45000") and convert to numeric
        temp_df = df[existing_cols].replace({',': ''}, regex=True).apply(pd.to_numeric, errors='coerce').fillna(0)
        majority = temp_df.idxmax(axis=1)
        
        # Map Hebrew/Raw names to clean English labels for the dashboard
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

def visualize_crime_rates_streamlit(merged_df, cpi_df=None):
    st.set_page_config(page_title="Israel Crime Stats", layout="wide")
    
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
    # VISUALIZATIONS (REORDERED)
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

# --- Boilerplate to run standalone ---
if __name__ == "__main__":
    if st.runtime.exists():
        # Update paths based on your actual local directories
        csv_path = "/Users/noamzelig/Desktop/Noam/Visualization_course/final_project/merged_crime_population_final.csv"
        cpi_path = "/Users/noamzelig/Desktop/Noam/Visualization_course/final_project/quarterly_cpi_chained.csv"
        
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            print(f"\n[INFO] Crime CSV loaded. Shape: {df.shape}")
            
            try:
                cpi_df = pd.read_csv(cpi_path)
                print(f"[INFO] CPI CSV loaded. Shape: {cpi_df.shape}\n")
            except FileNotFoundError:
                cpi_df = None
                print(f"[WARNING] CPI CSV not found at: {cpi_path}\n")

            visualize_crime_rates_streamlit(df, cpi_df)
        except FileNotFoundError:
            st.error(f"Please ensure the primary file exists at: {csv_path}")
    else:
        print("------------------------------------------------------------------")
        print("⚠️  STREAMLIT NOT DETECTED")
        print("   To view the interactive dashboard, you must run this script via the Streamlit CLI.")
        print("   Please run the following command in your terminal:")
        print(f"\n   streamlit run {os.path.basename(__file__) if '__file__' in locals() else 'crime_visualization.py'}")
        print("\n------------------------------------------------------------------")