# ============================================================
#  app.py  —  IPL Analytics Streamlit Dashboard
#
#  Run with:  streamlit run app.py
#
#  Folder structure expected:
#    ipl_analytics/
#    ├── app.py
#    ├── data_loader.py
#    ├── analytics.py
#    ├── charts.py
#    └── (your Cricsheet JSON folder path set in SIDEBAR)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_ipl_data, build_innings2
from analytics  import (
    build_win_prob_lookup, attach_win_prob,
    phase_analysis, team_analysis, wicket_analysis,
    momentum_analysis, venue_analysis, toss_analysis,
    scoring_analysis, season_trends,
    batting_records, bowling_records, fielding_records,
    team_score_records, match_records,
)
from charts import (
    chart_match_replayer, chart_phase_overview,
    chart_team_win_rates, chart_team_journeys,
    chart_wicket_impact, chart_venue,
    chart_toss, chart_scoring, chart_season_trends,
    chart_win_prob_heatmap,
    chart_phase_distribution,
    chart_team_performance_bubble,
    chart_venue_bubble,
    chart_season_flow,
    chart_turning_points_scatter,
    chart_record_barh,
    chart_record_barv,
)

# ============================================================
#  PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title  = "IPL Win Probability Analytics",
    page_icon   = "🏏",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ── GLOBAL STYLE ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Rajdhani:wght@400;500;600;700&display=swap');

  :root {
      --bg-1: #03112a;
      --bg-2: #07204a;
      --panel: rgba(8, 30, 63, 0.82);
      --panel-strong: rgba(9, 34, 71, 0.95);
      --text: #e7f0ff;
      --muted: #b7caec;
      --line: rgba(165, 198, 255, 0.24);
      --gold: #f6c453;
      --rose: #ff5d4a;
      --cyan: #3dc3ff;
      --lime: #7ce38b;
  }

    html, body, [class*="css"] {
            font-family: 'Rajdhani', sans-serif;
      color: var(--text);
  }

  [data-testid="stAppViewContainer"] {
      background:
                radial-gradient(920px 520px at 93% -8%, rgba(246,196,83,0.20), transparent 55%),
                radial-gradient(760px 460px at -8% 15%, rgba(61,195,255,0.20), transparent 52%),
                linear-gradient(145deg, var(--bg-1) 0%, #04193a 45%, var(--bg-2) 100%);
  }

  [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #061832 0%, #041229 100%);
      border-right: 1px solid var(--line);
  }
  @media (min-width: 1024px) {
      [data-testid="stSidebar"][aria-expanded="true"] {
          min-width: 320px;
          max-width: 320px;
      }
  }
  [data-testid="collapsedControl"] {
      display: flex !important;
      visibility: visible !important;
      opacity: 1 !important;
      z-index: 10000 !important;
  }
  [data-testid="stHeader"] { background: transparent; }

  .block-container {
      padding-top: 1rem;
      position: relative;
      z-index: 1;
  }

  .noise-overlay {
      position: fixed;
      inset: 0;
      pointer-events: none;
      z-index: 0;
      opacity: 0.06;
      background-image: radial-gradient(rgba(255,255,255,0.35) 0.45px, transparent 0.45px);
      background-size: 3px 3px;
      mix-blend-mode: soft-light;
  }

  .flare {
      position: fixed;
      width: 300px;
      height: 300px;
      border-radius: 999px;
      filter: blur(56px);
      opacity: 0.18;
      pointer-events: none;
      z-index: 0;
      animation: drift 11s ease-in-out infinite;
  }
    .flare-a { top: -80px; right: -90px; background: var(--gold); }
    .flare-b { bottom: -70px; left: -90px; background: var(--cyan); animation-delay: 1.6s; }
  @keyframes drift {
      0%, 100% { transform: translateY(0px) scale(1); }
      50% { transform: translateY(-15px) scale(1.05); }
  }

  h1, h2, h3, h4 {
      color: var(--text) !important;
      font-family: 'Bebas Neue', sans-serif;
      letter-spacing: 0.8px;
      font-weight: 400;
  }
    h1 { font-size: 3.45rem !important; line-height: 1.02; }
    h2 { font-size: 2.65rem !important; line-height: 1.06; }
    h3 { font-size: 2.2rem !important; line-height: 1.08; }
    h4 { font-size: 1.85rem !important; line-height: 1.12; }

  .hero {
      border: 1px solid rgba(246, 196, 83, 0.42);
      border-radius: 18px;
      padding: 18px 18px 16px 18px;
      margin: 8px 0 18px 0;
      background: linear-gradient(112deg, rgba(20,79,163,0.34), rgba(246,196,83,0.20) 46%, rgba(61,195,255,0.18));
      box-shadow: 0 16px 38px rgba(0,0,0,0.33);
      backdrop-filter: blur(6px);
  }
  .hero-title {
      font-family: 'Bebas Neue', sans-serif;
      letter-spacing: 0.9px;
    font-size: 2.25rem;
      line-height: 1;
      margin-bottom: 5px;
      color: #f9fcff;
  }
  .hero-sub {
      color: #d7e8ff;
      opacity: 0.94;
    font-size: 1.18rem;
      margin-bottom: 11px;
  }
  .hero-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
  }
  .meta-chip {
      border: 1px solid rgba(255,255,255,0.23);
      border-radius: 999px;
      padding: 4px 9px;
      font-size: 0.72rem;
      background: rgba(25, 19, 40, 0.62);
      color: #eef6ff;
  }

  .page-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    border: 1px solid rgba(246,196,83,0.35);
      border-radius: 14px;
      padding: 12px 14px;
      margin: 2px 0 14px 0;
      background: linear-gradient(102deg, rgba(20,79,163,0.34), rgba(246,196,83,0.20), rgba(61,195,255,0.14));
      box-shadow: 0 12px 30px rgba(0,0,0,0.26);
  }
  .page-title {
      font-family: 'Bebas Neue', sans-serif;
      letter-spacing: 0.7px;
    font-size: 2.05rem;
      line-height: 1;
      color: #f7fbff;
  }
  .page-sub {
      margin-top: 4px;
      color: #d8e8ff;
    font-size: 1.12rem;
  }
  .live-pill {
      font-size: 0.72rem;
      font-weight: 700;
      letter-spacing: 0.5px;
      text-transform: uppercase;
      border-radius: 999px;
      padding: 6px 10px;
      border: 1px solid rgba(139, 247, 122, 0.55);
      background: rgba(139, 247, 122, 0.16);
      color: #e8ffe3;
      box-shadow: 0 0 0 0 rgba(139,247,122,0.5);
      animation: pulse-live 1.7s infinite;
      white-space: nowrap;
  }
  @keyframes pulse-live {
      0% { box-shadow: 0 0 0 0 rgba(139,247,122,0.45); }
      70% { box-shadow: 0 0 0 10px rgba(139,247,122,0); }
      100% { box-shadow: 0 0 0 0 rgba(139,247,122,0); }
  }

  .metric-card {
      border: 1px solid rgba(165,198,255,0.26);
      border-radius: 14px;
      padding: 15px 14px;
      text-align: center;
    background: linear-gradient(180deg, rgba(7,35,73,0.94), rgba(5,25,55,0.95));
      box-shadow: 0 10px 24px rgba(0,0,0,0.3);
      transition: transform .16s ease, border-color .16s ease;
  }
  .metric-card:hover {
      transform: translateY(-2px);
      border-color: rgba(255,180,83,0.42);
  }
  .metric-value {
      font-family: 'Bebas Neue', sans-serif;
      letter-spacing: 0.7px;
      font-size: 2rem;
      line-height: 1;
      margin-bottom: 2px;
      background: linear-gradient(90deg, var(--gold), #ffd977);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
  }
  .metric-label {
    font-size: 1.02rem;
      letter-spacing: 0.3px;
      color: var(--muted);
      text-transform: uppercase;
  }

  .loading-overlay {
      position: fixed;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 9999;
      background: radial-gradient(circle at center, rgba(3, 17, 42, 0.20), rgba(3, 17, 42, 0.62));
      backdrop-filter: blur(4px);
  }
  .loading-card {
      display: flex;
      align-items: center;
      gap: 18px;
      padding: 18px 24px;
      border-radius: 18px;
      border: 1px solid rgba(246,196,83,0.30);
      background: linear-gradient(135deg, rgba(7, 35, 73, 0.92), rgba(5, 25, 55, 0.95));
      box-shadow: 0 16px 40px rgba(0,0,0,0.35), 0 0 0 1px rgba(255,255,255,0.04) inset;
  }
  .ball-loader {
      position: relative;
      width: 46px;
      height: 46px;
      border-radius: 50%;
      background: radial-gradient(circle at 32% 30%, #ff9a9a 0%, #ff4b4b 38%, #cf1d1d 100%);
      box-shadow: 0 0 20px rgba(255,75,75,0.35);
      animation: ballBounce 1.1s ease-in-out infinite, ballSpin 1.4s linear infinite;
      flex-shrink: 0;
  }
  .ball-loader::before,
  .ball-loader::after {
      content: '';
      position: absolute;
      inset: 0;
      border-radius: 50%;
      border: 2px solid rgba(255,255,255,0.72);
      clip-path: polygon(49% 0%, 54% 0%, 60% 8%, 66% 18%, 70% 30%, 72% 44%, 73% 58%, 71% 72%, 66% 84%, 59% 94%, 53% 100%, 48% 100%, 43% 93%, 37% 83%, 32% 71%, 29% 57%, 28% 44%, 30% 30%, 34% 18%, 40% 8%);
      opacity: 0.9;
  }
  .ball-loader::after {
      transform: rotate(180deg);
  }
  @keyframes ballBounce {
      0%, 100% { transform: translateY(0) scale(1); }
      50% { transform: translateY(-7px) scale(1.03); }
  }
  @keyframes ballSpin {
      0% { filter: hue-rotate(0deg); }
      100% { filter: hue-rotate(0deg); }
  }
  .loading-copy {
      display: flex;
      flex-direction: column;
      gap: 4px;
      color: #eef6ff;
  }
  .loading-title {
      font-family: 'Bebas Neue', sans-serif;
      font-size: 1.55rem;
      letter-spacing: 0.8px;
      line-height: 1;
  }
  .loading-sub {
      font-size: 1rem;
      color: #d0e1ff;
      letter-spacing: 0.2px;
  }

  .section-divider {
      border-top: 1px solid rgba(255,255,255,0.14);
      margin: 1.5rem 0;
  }

  .stTextInput > div > div > input,
  .stSelectbox > div > div,
  .stMultiSelect > div > div,
  .stSlider > div > div,
  .stNumberInput > div > div > input {
      border-radius: 10px !important;
      border: 1px solid rgba(255,255,255,0.2) !important;
      background: rgba(28, 21, 44, 0.86) !important;
      color: #e7f0ff !important;
  }

  /* BaseWeb selectors make widgets consistent across Streamlit versions */
  div[data-baseweb="select"] > div {
      border-radius: 10px !important;
      border: 1px solid rgba(255,255,255,0.22) !important;
      background: rgba(10, 30, 62, 0.94) !important;
      color: #e7f0ff !important;
  }
  div[data-baseweb="select"] input,
  div[data-baseweb="select"] span,
  div[data-baseweb="select"] svg,
  div[data-baseweb="select"] div {
      color: #e7f0ff !important;
  }
  div[data-baseweb="popover"] {
      background: rgba(8, 26, 56, 0.98) !important;
      border: 1px solid rgba(130, 185, 255, 0.25) !important;
  }
  div[data-baseweb="popover"] li,
  div[data-baseweb="popover"] div {
      color: #e7f0ff !important;
      background: transparent !important;
  }
  [data-baseweb="tag"] {
      background: #ff5d4a !important;
      color: #ffffff !important;
      border-radius: 8px !important;
  }

  [data-testid="stSidebar"] * {
      color: #e7f0ff;
  }

  .stRadio [role="radiogroup"] {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      padding: 12px;
      border: 1px solid rgba(130, 185, 255, 0.28);
      border-radius: 16px;
      background:
          linear-gradient(115deg, rgba(8, 33, 69, 0.88), rgba(5, 22, 50, 0.82)),
          radial-gradient(600px 160px at 20% -30%, rgba(246, 196, 83, 0.12), transparent 60%);
      box-shadow: 0 10px 26px rgba(0, 0, 0, 0.25), inset 0 0 0 1px rgba(255,255,255,0.03);
      backdrop-filter: blur(4px);
  }
  .stRadio [role="radiogroup"] > label {
      border: 1px solid rgba(148, 191, 255, 0.20);
      border-radius: 12px;
      padding: 9px 13px;
      min-height: 40px;
      background: linear-gradient(180deg, rgba(14, 50, 98, 0.68), rgba(8, 34, 71, 0.66));
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.06);
      transition: transform .18s ease, border-color .18s ease, background .18s ease, box-shadow .18s ease;
  }
  .stRadio [role="radiogroup"] > label:hover {
      transform: translateY(-1px);
      border-color: rgba(246,196,83,0.52);
      background: linear-gradient(180deg, rgba(25, 72, 133, 0.78), rgba(10, 41, 83, 0.74));
      box-shadow: 0 6px 16px rgba(0,0,0,0.24), 0 0 0 1px rgba(246,196,83,0.2) inset;
  }
  .stRadio [role="radiogroup"] > label:has(input:checked) {
      border-color: rgba(246,196,83,0.80);
      background: linear-gradient(120deg, rgba(246,196,83,0.22), rgba(61,195,255,0.24));
      box-shadow: 0 8px 20px rgba(5, 20, 43, 0.35), 0 0 18px rgba(246,196,83,0.20);
  }
  .stRadio [role="radiogroup"] > label > div {
    font-size: 1.24rem;
      font-weight: 600;
      letter-spacing: 0.15px;
  }

  button[kind="primary"] {
      border: none !important;
      border-radius: 11px !important;
      color: #082247 !important;
      font-weight: 700 !important;
      background: linear-gradient(90deg, var(--gold), #ffe39b) !important;
  }

  [data-testid="stMetric"] {
      border: 1px solid rgba(255,255,255,0.17);
      border-radius: 12px;
      background: rgba(29, 22, 46, 0.84);
      padding: 12px 14px;
  }
  [data-testid="stDataFrame"] {
      border: 1px solid rgba(255,255,255,0.16);
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 8px 18px rgba(0,0,0,0.2);
  }

  [data-baseweb="tab-list"] {
      border: 1px solid rgba(255,255,255,0.18);
      border-radius: 14px;
      background: rgba(24,18,39,0.86);
      gap: 0;
      padding: 8px 10px;
  }
  [data-baseweb="tab"] {
      border-radius: 10px;
      min-height: 52px;
      padding: 10px 18px !important;
    font-size: 1.42rem;
      font-weight: 600;
      letter-spacing: 0.1px;
      margin: 0;
  }
  [data-baseweb="tab"]:not(:last-child) {
      border-right: 1px solid rgba(255,255,255,0.14);
      margin-right: 6px;
      padding-right: 20px !important;
  }
  [data-baseweb="tab-highlight"] {
      border-radius: 10px;
      background: linear-gradient(90deg, rgba(246,196,83,0.30), rgba(61,195,255,0.28));
  }

  [data-testid="stSidebar"] .block-container {
      padding-top: 0.55rem;
      padding-bottom: 0.7rem;
  }
  .sidebar-brand {
      padding: 14px 14px;
      border: 1px solid rgba(174,191,236,.24);
      border-radius: 14px;
    background: linear-gradient(180deg, rgba(8,37,74,.82), rgba(6,28,58,.78));
      margin-bottom: 4px;
      box-shadow: 0 8px 18px rgba(0,0,0,.24);
  }
  .sidebar-brand-title {
      display: flex;
      align-items: center;
      gap: 10px;
      font-weight: 700;
      font-size: 1.28rem;
      color: #f0f3ff;
      line-height: 1;
  }
  .sidebar-brand-icon {
      font-size: 1.55rem;
      line-height: 1;
      filter: drop-shadow(0 0 10px rgba(246,196,83,.45));
  }
  .sidebar-brand-sub {
    font-size: 1.05rem;
      color: #aeb9de;
      margin-top: 7px;
  }
  .sidebar-filters-title {
      margin-top: 2px;
      margin-bottom: 6px;
    font-size: 1.34rem;
      font-weight: 700;
      color: #f6e8f1;
  }

  /* Dataframe headers and cells */
  [data-testid="stDataFrame"] thead th {
    font-size: 1.08rem !important;
      font-weight: 700 !important;
  }
  [data-testid="stDataFrame"] tbody td {
    font-size: 1rem !important;
  }

  /* Widget labels */
  [data-testid="stWidgetLabel"] {
      font-size: 1.12rem !important;
      font-weight: 600 !important;
  }

  [data-testid="stTooltipIcon"] {
      display: none !important;
  }
  div[data-testid="stSpinner"] {
      padding: 1rem 0 0.5rem 0;
  }
  div[data-testid="stSpinner"] > div {
      border-color: rgba(246,196,83,0.35) rgba(61,195,255,0.30) rgba(246,196,83,0.35) rgba(61,195,255,0.30) !important;
      width: 2.3rem !important;
      height: 2.3rem !important;
  }
  [data-testid="stToolbar"] {
      display: flex !important;
      visibility: visible !important;
      opacity: 1 !important;
  }
  [data-testid="stStatusWidget"] {
      display: none !important;
  }
  #MainMenu {
      visibility: hidden;
  }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='noise-overlay'></div><div class='flare flare-a'></div><div class='flare flare-b'></div>", unsafe_allow_html=True)


# Prefer local dataset path inside this project.
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "ipl_ball_by_ball"


def render_page_header(title: str, subtitle: str):
        st.markdown(
                f"""
                <div class='page-head'>
                    <div>
                        <div class='page-title'>{title}</div>
                        <div class='page-sub'>{subtitle}</div>
                    </div>
                    <div class='live-pill'>Live Data</div>
                </div>
                """,
                unsafe_allow_html=True,
        )


# ============================================================
#  DATA LOADING (CACHED)
# ============================================================
@st.cache_data(show_spinner="Loading IPL data...")
def load_all(path: str):
    ball_df, match_df    = load_ipl_data(path)
    inn2_df              = build_innings2(ball_df)
    lookup, binned       = build_win_prob_lookup(inn2_df)
    df                   = attach_win_prob(inn2_df, lookup, binned)

    # Pre-compute all analytics
    phase_data   = phase_analysis(df)
    team_data    = team_analysis(df)
    wicket_data  = wicket_analysis(df)
    momentum_data = momentum_analysis(df)
    venue_data   = venue_analysis(df, match_df)
    toss_data    = toss_analysis(match_df)
    scoring_data = scoring_analysis(ball_df)
    season_data  = season_trends(match_df, ball_df)

    return {
        'ball_df'      : ball_df,
        'match_df'     : match_df,
        'inn2_df'      : inn2_df,
        'df'           : df,           # 2nd innings with win_prob
        'phase'        : phase_data,
        'team'         : team_data,
        'wicket'       : wicket_data,
        'momentum'     : momentum_data,
        'venue'        : venue_data,
        'toss'         : toss_data,
        'scoring'      : scoring_data,
        'season'       : season_data,
    }


# ============================================================
#  SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
        <div class='sidebar-brand'>
            <div class='sidebar-brand-title'>
                <span class='sidebar-brand-icon'>🏏</span>
                <span>IPL Analytics Studio</span>
            </div>
            <div class='sidebar-brand-sub'>Interactive chase intelligence dashboard</div>
    </div>
    """, unsafe_allow_html=True)

data_path = str(DEFAULT_DATA_PATH)


# ============================================================
#  LOAD DATA
# ============================================================
if not os.path.isdir(data_path):
    st.error(f"❌ Folder not found: `{data_path}`  \nPlease enter a valid path in the sidebar.")
    st.stop()

json_count = len(list(Path(data_path).glob("*.json")))
if json_count == 0:
    st.error(
        f"❌ No JSON files found in `{data_path}`.\n"
        "Please point to the extracted Cricsheet IPL ball-by-ball JSON folder."
    )
    st.stop()

loading_overlay = st.empty()
loading_overlay.markdown(
    """
    <div class='loading-overlay'>
        <div class='loading-card'>
            <div class='ball-loader'></div>
            <div class='loading-copy'>
                <div class='loading-title'>LOADING IPL DATA</div>
                <div class='loading-sub'>Buffering the scorecards, charts, and records</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    data = load_all(data_path)
except Exception as e:
    st.error(f"❌ Failed to load data: {e}")
    st.stop()
loading_overlay.empty()

raw_df       = data['df']
raw_match_df = data['match_df']
raw_ball_df  = data['ball_df']

with st.sidebar:
    st.markdown("<div class='sidebar-filters-title'>🎛️ Dashboard Filters</div>", unsafe_allow_html=True)

    season_options = sorted(raw_match_df['season'].dropna().astype(str).unique().tolist())
    selected_seasons = st.multiselect(
        "Season(s)",
        options=season_options,
        default=season_options,
    )

    team_options = sorted([
        t for t in pd.unique(raw_match_df[['team1', 'team2']].values.ravel())
        if pd.notna(t)
    ])
    selected_teams = st.multiselect(
        "Team(s) (optional)",
        options=team_options,
        default=[],
    )

    venue_min_matches = st.slider("Min venue sample", min_value=5, max_value=20, value=8)

filtered_match_df = raw_match_df.copy()

if selected_seasons:
    filtered_match_df = filtered_match_df[
        filtered_match_df['season'].astype(str).isin(selected_seasons)
    ]

if selected_teams:
    filtered_match_df = filtered_match_df[
        filtered_match_df['team1'].isin(selected_teams) |
        filtered_match_df['team2'].isin(selected_teams)
    ]

if filtered_match_df.empty:
    st.warning("No matches found for selected filters. Please adjust sidebar filters.")
    st.stop()

filtered_match_ids = set(filtered_match_df['match_id'])
filtered_ball_df = raw_ball_df[raw_ball_df['match_id'].isin(filtered_match_ids)].copy()
filtered_df = raw_df[raw_df['match_id'].isin(filtered_match_ids)].copy()

if filtered_df.empty:
    st.warning("Filtered selection has no 2nd innings data. Please adjust filters.")
    st.stop()

view_data = {
    'ball_df': filtered_ball_df,
    'match_df': filtered_match_df,
    'df': filtered_df,
    'phase': phase_analysis(filtered_df),
    'team': team_analysis(filtered_df),
    'wicket': wicket_analysis(filtered_df),
    'momentum': momentum_analysis(filtered_df),
    'venue': venue_analysis(filtered_df, filtered_match_df),
    'toss': toss_analysis(filtered_match_df),
    'scoring': scoring_analysis(filtered_ball_df),
    'season': season_trends(filtered_match_df, filtered_ball_df),
    'batting': batting_records(filtered_ball_df, filtered_match_df),
    'bowling': bowling_records(filtered_ball_df, filtered_match_df),
    'fielding': fielding_records(filtered_ball_df),
    'team_scores': team_score_records(filtered_ball_df, filtered_match_df),
    'match_records': match_records(filtered_match_df, filtered_ball_df),
}

if not view_data['venue'].empty:
    view_data['venue'] = view_data['venue'][view_data['venue']['matches'] >= venue_min_matches]

df = view_data['df']
match_df = view_data['match_df']
ball_df = view_data['ball_df']

active_filter_badges = [f"Matches: {match_df['match_id'].nunique()}", f"Deliveries: {len(ball_df):,}"]
if selected_seasons and len(selected_seasons) != len(season_options):
    active_filter_badges.append(f"Seasons: {len(selected_seasons)}")
if selected_teams:
    active_filter_badges.append(f"Teams: {len(selected_teams)}")

st.markdown(
    """
    <div class='hero'>
        <div class='hero-title'>IPL Ball-by-Ball Intelligence Center</div>
        <div class='hero-sub'>Interactive, filter-aware performance cockpit with richer visual analytics.</div>
        <div class='hero-meta'>
            """ + "".join([f"<span class='meta-chip'>{badge}</span>" for badge in active_filter_badges]) + """
            <span class='meta-chip'>Chart Modes: 16+</span>
            <span class='meta-chip'>Model: Historical Outcomes Only</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

page_options = {
    "🏟️ Command Center": "overview",
    "🎬 Match Lab": "match_replayer",
    "🌊 Phase Flow": "phase_analysis",
    "🧭 Team Atlas": "team_profiles",
    "🎯 Wicket Shock": "wicket_impact",
    "🗺️ Venue Map": "venue_analysis",
    "🪙 Toss IQ": "toss_analysis",
    "💥 Scoring Engine": "scoring_patterns",
    "📅 Season Pulse": "season_trends",
    "🔥 State Heatmap": "win_prob_heatmap",
    "⚡ Swing Radar": "turning_points",
    "🏏 Batting Records": "batting_records",
    "🎳 Bowling Records": "bowling_records",
    "🧤 Fielding Records": "fielding_records",
    "📊 Team Scores": "team_scores",
    "🏅 Team & Captain Records": "team_captain_records",
}

selected_page = st.radio(
    "Sections",
    options=list(page_options.keys()),
    horizontal=True,
    label_visibility="collapsed",
)
page = page_options[selected_page]


# ============================================================
#  PAGE ROUTING
# ============================================================

# ── OVERVIEW ─────────────────────────────────────────────────
if page == "overview":
    render_page_header("Overview Dashboard", "Ball-by-ball historical analysis with zero model assumptions")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # KPI row
    total_matches   = match_df['match_id'].nunique()
    total_balls     = len(ball_df)
    seasons         = match_df['season'].dropna().astype(str).str.strip().nunique()
    teams           = match_df['team1'].nunique()
    chaser_wr       = view_data['season']['chaser_win_rate'].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, lbl in [
        (c1, total_matches, "Total Matches"),
        (c2, f"{total_balls:,}", "Total Deliveries"),
        (c3, seasons, "IPL Seasons"),
        (c4, teams, "Unique Teams"),
        (c5, f"{chaser_wr:.1%}", "Overall Chaser Win Rate"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-value">{val}</div>
          <div class="metric-label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    left, right = st.columns([1.3, 1])
    with left:
        st.subheader("Season Trends at a Glance")
        fig = chart_season_trends(view_data['season'])
        st.pyplot(fig, use_container_width=True)
    with right:
        st.subheader("Season Flow")
        fig_flow = chart_season_flow(view_data['season'])
        st.pyplot(fig_flow, use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("Win Probability Heatmap — Overall")
    st.caption("How often does the chasing team win from each game state? (Pure historical win%)")
    fig2 = chart_win_prob_heatmap(df)
    st.pyplot(fig2, use_container_width=True)


# ── MATCH REPLAYER ────────────────────────────────────────────
elif page == "match_replayer":
    render_page_header("Ball-by-Ball Match Replayer", "Win probability reflects historical outcomes for identical game states")

    col_l, col_r = st.columns([1, 3])

    with col_l:
        # Build match labels for dropdown
        mdf = match_df.copy()
        mdf['label'] = (
            mdf['team1'] + " vs " + mdf['team2'] +
            "  |  " + mdf['season'].astype(str)
        )
        match_labels = dict(zip(mdf['label'], mdf['match_id']))
        selected_label = st.selectbox("Select a match", sorted(match_labels.keys()))
        selected_id    = match_labels[selected_label]

        match_row = match_df[match_df['match_id'] == selected_id].iloc[0]
        st.markdown(f"""
        <div style='background:#1a1d27; border:1px solid #2e3150;
                    border-radius:8px; padding:14px; margin-top:12px;
                    font-size:0.85rem; line-height:1.8; color:#c8cde4'>
        🏆 <b style='color:#7c6af7'>Winner:</b> {match_row['winner']}<br>
        🏟️ <b style='color:#7c6af7'>Venue:</b> {match_row.get('venue','—')}<br>
        📅 <b style='color:#7c6af7'>Date:</b> {match_row.get('date','—')}<br>
        🎲 <b style='color:#7c6af7'>Toss:</b> {match_row.get('toss_winner','—')} elected to {match_row.get('toss_decision','—')}
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        match_balls = df[df['match_id'] == selected_id]
        if len(match_balls) == 0:
            st.warning("No 2nd innings data for this match.")
        else:
            fig = chart_match_replayer(match_balls)
            st.pyplot(fig, use_container_width=True)

    # Key moments table
    st.markdown("---")
    st.subheader("Key Moments — Biggest Win Prob Swings in This Match")
    mb = df[df['match_id'] == selected_id].copy()
    if 'swing' in mb.columns:
        top_moments = (
            mb.dropna(subset=['swing'])
            .nlargest(8, 'swing')
            [['over', 'legal_ball_in_inn', 'batter', 'bowler',
              'runs_total', 'is_wicket', 'win_prob_prev', 'win_prob', 'win_prob_delta']]
            .rename(columns={
                'over': 'Over', 'legal_ball_in_inn': 'Ball',
                'batter': 'Batter', 'bowler': 'Bowler',
                'runs_total': 'Runs', 'is_wicket': 'Wicket?',
                'win_prob_prev': 'Prob Before', 'win_prob': 'Prob After',
                'win_prob_delta': 'Δ Prob'
            })
        )
        for c in ['Prob Before', 'Prob After', 'Δ Prob']:
            top_moments[c] = top_moments[c].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
        top_moments['Wicket?'] = top_moments['Wicket?'].map({0: '', 1: '🔴 Yes'})
        st.dataframe(top_moments, use_container_width=True, hide_index=True)


# ── PHASE ANALYSIS ────────────────────────────────────────────
elif page == "phase_analysis":
    render_page_header("Phase Analysis", "Track pressure and volatility across Powerplay, Middle, and Death overs")

    t1, t2 = st.tabs(["Phase Trend", "Phase Distribution"])

    with t1:
        fig = chart_phase_overview(view_data['phase'])
        st.pyplot(fig, use_container_width=True)

    with t2:
        fig_dist = chart_phase_distribution(df)
        st.pyplot(fig_dist, use_container_width=True)

    st.markdown("---")
    st.subheader("Phase Summary Table")
    phase_tbl = view_data['phase']['by_phase'].copy()
    for c in ['avg_win_prob', 'avg_swing', 'avg_pressure']:
        if c in phase_tbl.columns:
            phase_tbl[c] = phase_tbl[c].map(lambda x: f"{x:.4f}")
    st.dataframe(phase_tbl, use_container_width=True, hide_index=True)


# ── TEAM PROFILES ─────────────────────────────────────────────
elif page == "team_profiles":
    render_page_header("Team Chase Profiles", "Compare team-level chase stability, win rates, and journey curves")

    tab1, tab2, tab3 = st.tabs(["Win Rates & Volatility", "Journey Comparison", "Performance Bubble"])

    with tab1:
        fig = chart_team_win_rates(view_data['team']['stats'])
        st.pyplot(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Full Team Stats Table")
        ts = view_data['team']['stats'].copy()
        for c in ['win_rate', 'avg_win_prob', 'volatility', 'avg_pressure']:
            if c in ts.columns:
                ts[c] = ts[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
        st.dataframe(ts, use_container_width=True, hide_index=True)

    with tab2:
        all_teams = sorted(view_data['team']['stats']['batting_team'].unique())
        selected_teams = st.multiselect(
            "Select teams to compare (max 8)",
            all_teams,
            default=all_teams[:6],
            max_selections=8
        )
        if selected_teams:
            fig2 = chart_team_journeys(view_data['team']['journey'], selected_teams)
            st.pyplot(fig2, use_container_width=True)
        else:
            st.info("Select at least one team.")

    with tab3:
        fig3 = chart_team_performance_bubble(view_data['team']['stats'])
        st.pyplot(fig3, use_container_width=True)


# ── WICKET IMPACT ─────────────────────────────────────────────
elif page == "wicket_impact":
    render_page_header("Wicket Impact Analysis", "See which wicket number causes the largest average win-probability shock")

    fig = chart_wicket_impact(view_data['wicket'])
    st.pyplot(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Wicket Impact Table")
    wi = view_data['wicket'].copy()
    for c in ['avg_prob_before', 'avg_prob_after', 'prob_drop']:
        wi[c] = wi[c].map(lambda x: f"{x:.3f}")
    st.dataframe(wi.rename(columns={
        'wicket_number': 'Wicket #',
        'avg_prob_before': 'Avg Prob Before',
        'avg_prob_after' : 'Avg Prob After',
        'prob_drop'      : 'Avg Drop',
        'count'          : 'Sample Size'
    }), use_container_width=True, hide_index=True)


# ── VENUE ANALYSIS ────────────────────────────────────────────
elif page == "venue_analysis":
    render_page_header("Venue Analysis", "Identify the most chase-friendly and bowling-friendly stadiums")

    top_n = st.slider("Number of venues to show", 8, 20, 12)
    if view_data['venue'].empty:
        st.warning("No venue data after applying filters. Lower the Min venue sample in sidebar.")
    else:
        v1, v2 = st.columns(2)
        with v1:
            fig = chart_venue(view_data['venue'], top_n=top_n)
            st.pyplot(fig, use_container_width=True)
        with v2:
            fig_b = chart_venue_bubble(view_data['venue'])
            st.pyplot(fig_b, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Most Chaser-Friendly")
        st.dataframe(
            view_data['venue'][['venue', 'matches', 'chaser_win_rate']].head(5)
            .assign(**{'chaser_win_rate': lambda x: x['chaser_win_rate'].map('{:.1%}'.format)}),
            use_container_width=True, hide_index=True
        )
    with col2:
        st.subheader("Most Bowling-Friendly")
        st.dataframe(
            view_data['venue'][['venue', 'matches', 'chaser_win_rate']].tail(5)
            .assign(**{'chaser_win_rate': lambda x: x['chaser_win_rate'].map('{:.1%}'.format)}),
            use_container_width=True, hide_index=True
        )


# ── TOSS ANALYSIS ─────────────────────────────────────────────
elif page == "toss_analysis":
    render_page_header("Toss Analysis", "Understand toss strategy patterns and match-win conversion impact")

    wr = view_data['toss']['overall_toss_win_rate']
    st.metric("Overall: toss winner also won the match", f"{wr:.1%}")

    fig = chart_toss(view_data['toss'])
    st.pyplot(fig, use_container_width=True)


# ── SCORING PATTERNS ──────────────────────────────────────────
elif page == "scoring_patterns":
    render_page_header("Scoring Patterns", "Boundary frequency, six rates, dot-ball pressure, and over-wise scoring")

    sd = view_data['scoring']
    c1, c2, c3 = st.columns(3)
    c1.metric("Boundary Ball %", f"{sd['boundary_rate']:.1%}")
    c2.metric("Six Ball %",      f"{sd['six_rate']:.1%}")
    c3.metric("Dot Ball %",      f"{sd['dot_rate']:.1%}")

    fig = chart_scoring(sd)
    st.pyplot(fig, use_container_width=True)


# ── SEASON TRENDS ─────────────────────────────────────────────
elif page == "season_trends":
    render_page_header("Season Trends", "Evolution of first-innings scores and chase success by IPL season")

    s1, s2 = st.columns(2)
    with s1:
        fig = chart_season_trends(view_data['season'])
        st.pyplot(fig, use_container_width=True)
    with s2:
        fig2 = chart_season_flow(view_data['season'])
        st.pyplot(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Season Data Table")
    st.dataframe(
        view_data['season']
        .assign(**{
            'chaser_win_rate': lambda x: x['chaser_win_rate'].map('{:.1%}'.format),
            'avg_first_innings_score': lambda x: x['avg_first_innings_score'].map('{:.1f}'.format),
        }),
        use_container_width=True, hide_index=True
    )


# ── WIN PROB HEATMAP ──────────────────────────────────────────
elif page == "win_prob_heatmap":
    render_page_header("Win Probability Heatmap", "State-space map of wickets left versus balls left across IPL chases")
    st.caption("""
    Each cell = historical win% for the chasing team in that game state.
    **No model. No prediction.** Just: out of all matches where the chasing team
    had X wickets left and Y balls left, how many did they win?
    """)

    fig = chart_win_prob_heatmap(df)
    st.pyplot(fig, use_container_width=True)


# ── TURNING POINTS ────────────────────────────────────────────
elif page == "turning_points":
    render_page_header("Biggest Turning Points", "Single deliveries causing the largest win-probability swing")

    fig_swing = chart_turning_points_scatter(view_data['momentum'])
    st.pyplot(fig_swing, use_container_width=True)
    st.markdown("---")

    top = view_data['momentum'].copy()
    top['win_prob_delta'] = top['win_prob_delta'].map(lambda x: f"{x:+.3f}" if pd.notna(x) else "—")
    top['win_prob_prev']  = top['win_prob_prev'].map(lambda x: f"{x:.2%}"   if pd.notna(x) else "—")
    top['win_prob']       = top['win_prob'].map(lambda x: f"{x:.2%}"        if pd.notna(x) else "—")
    top['is_wicket']      = top['is_wicket'].map({0: '', 1: '🔴'})
    top['swing']          = top['swing'].map(lambda x: f"{x:.4f}")

    st.dataframe(top.rename(columns={
        'match_id'         : 'Match',
        'batting_team'     : 'Batting Team',
        'over'             : 'Over',
        'legal_ball_in_inn': 'Ball',
        'batter'           : 'Batter',
        'bowler'           : 'Bowler',
        'runs_total'       : 'Runs',
        'is_wicket'        : 'Wicket?',
        'win_prob_prev'    : 'Prob Before',
        'win_prob'         : 'Prob After',
        'win_prob_delta'   : 'Δ Prob',
        'swing'            : 'Swing',
        'match_phase'      : 'Phase',
    }), use_container_width=True, hide_index=True)


# ── BATTING RECORDS ──────────────────────────────────────────
elif page == "batting_records":
    render_page_header("Batting Records", "Runs, boundaries, consistency markers, and multi-team player insights")
    b = view_data['batting']

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tracked Batters", f"{len(b['runs'])}")
    m2.metric("Total Runs", f"{int(b['runs']['total_runs'].sum()):,}" if not b['runs'].empty else "0")
    m3.metric("Total Sixes", f"{int(b['sixes']['sixes'].sum()):,}" if not b['sixes'].empty else "0")
    m4.metric("Total Fours", f"{int(b['fours']['fours'].sum()):,}" if not b['fours'].empty else "0")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Run Machines", "Boundary Hitters", "Singles & Dot Pressure", "Milestones", "Multi-Team Players"
    ])

    with tab1:
        st.subheader("Run Leaders")
        st.caption("Career run aggregates across selected filters.")
        n = st.slider("Top batters", 5, 30, 15, key="bat_runs_n")
        st.pyplot(
            chart_record_barh(b['runs'], 'total_runs', 'batter', 'Most Runs in IPL', 'Total Runs', n, '#7c6af7'),
            use_container_width=True,
        )
        st.dataframe(b['runs'].head(n), use_container_width=True, hide_index=True, height=360)

    with tab2:
        st.subheader("Power Hit Spectrum")
        st.caption("Separated views for sixes, fours, and total boundaries.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Most Sixes")
            st.pyplot(chart_record_barh(b['sixes'], 'sixes', 'batter', 'Most Sixes', 'Sixes', 12, '#f0c84a'), use_container_width=True)
            st.dataframe(b['sixes'].head(12), use_container_width=True, hide_index=True, height=280)
        with c2:
            st.markdown("##### Most Fours")
            st.pyplot(chart_record_barh(b['fours'], 'fours', 'batter', 'Most Fours', 'Fours', 12, '#5cdb95'), use_container_width=True)
            st.dataframe(b['fours'].head(12), use_container_width=True, hide_index=True, height=280)

        st.markdown("##### Combined Boundary Count")
        st.pyplot(chart_record_barh(b['boundaries'], 'boundaries', 'batter', 'Most Boundaries', 'Boundaries', 15, '#3ec6a0'), use_container_width=True)
        st.dataframe(b['boundaries'].head(15), use_container_width=True, hide_index=True, height=320)

    with tab3:
        st.subheader("Rotation vs Pressure")
        st.caption("Compare run rotation ability (singles) with dot-ball pressure.")
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(chart_record_barh(b['singles'], 'singles', 'batter', 'Most Singles', 'Singles', 15, '#f0824a'), use_container_width=True)
            st.dataframe(b['singles'].head(15), use_container_width=True, hide_index=True, height=300)
        with c2:
            st.pyplot(chart_record_barh(b['dots_faced'], 'dot_balls_faced', 'batter', 'Most Dot Balls Faced', 'Dot Balls Faced', 15, '#4a4f6e'), use_container_width=True)
            st.dataframe(b['dots_faced'].head(15), use_container_width=True, hide_index=True, height=300)

    with tab4:
        st.subheader("Milestones & Fragility")
        st.caption("Fifties and hundreds measure conversion; ducks show early-dismissal risk.")
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(chart_record_barh(b['fifties'], 'fifties', 'batter', 'Most Fifties', '50s', 12, '#3ec6a0'), use_container_width=True)
            st.dataframe(b['fifties'].head(12), use_container_width=True, hide_index=True, height=270)
        with c2:
            st.pyplot(chart_record_barh(b['hundreds'], 'hundreds', 'batter', 'Most Hundreds', '100s', 12, '#f0c84a'), use_container_width=True)
            st.dataframe(b['hundreds'].head(12), use_container_width=True, hide_index=True, height=270)

        st.pyplot(chart_record_barh(b['ducks'], 'ducks', 'batter', 'Most Ducks', 'Ducks', 15, '#e05c6f'), use_container_width=True)
        st.dataframe(b['ducks'].head(15), use_container_width=True, hide_index=True, height=300)

    with tab5:
        st.subheader("Multi-Team Players")
        st.caption("Batters who represented multiple IPL franchises in selected filters.")
        st.dataframe(b['multi_team'], use_container_width=True, hide_index=True, height=420)


# ── BOWLING RECORDS ──────────────────────────────────────────
elif page == "bowling_records":
    render_page_header("Bowling Records", "Wickets, economy, dot-ball pressure, and best bowling figures")
    bw = view_data['bowling']

    bm1, bm2, bm3, bm4 = st.columns(4)
    bm1.metric("Tracked Bowlers", f"{len(bw['wickets'])}")
    bm2.metric("Total Wickets", f"{int(bw['wickets']['wickets'].sum()):,}" if not bw['wickets'].empty else "0")
    bm3.metric("Total Dot Balls", f"{int(bw['dots_bowled']['dot_balls_bowled'].sum()):,}" if not bw['dots_bowled'].empty else "0")
    bm4.metric("Best Economy", f"{bw['economy']['economy'].min():.2f}" if not bw['economy'].empty else "—")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Wickets & Economy", "Dot Ball Masters", "Best Figures", "Season Leaders"
    ])

    with tab1:
        st.subheader("Strike & Control")
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(chart_record_barh(bw['wickets'], 'wickets', 'bowler', 'Most Wickets', 'Wickets', 15, '#e05c6f'), use_container_width=True)
            st.dataframe(bw['wickets'].head(15), use_container_width=True, hide_index=True, height=300)
        with c2:
            st.caption("Economy chart uses bowlers with >= 10 overs")
            st.pyplot(chart_record_barh(bw['economy'], 'economy', 'bowler', 'Best Economy Rates', 'Economy', 15, '#3ec6a0'), use_container_width=True)
            st.dataframe(bw['economy'].head(15)[['bowler', 'overs', 'runs_conceded', 'economy']].round(2), use_container_width=True, hide_index=True, height=300)

    with tab2:
        st.subheader("Dot Ball Masters")
        st.caption("Sustained pressure creators through legal dot deliveries.")
        st.pyplot(chart_record_barh(bw['dots_bowled'], 'dot_balls_bowled', 'bowler', 'Most Dot Balls Bowled', 'Dot Balls', 20, '#7c6af7'), use_container_width=True)
        st.dataframe(bw['dots_bowled'].head(20), use_container_width=True, hide_index=True, height=420)

    with tab3:
        st.subheader("Best Figures in an Innings")
        st.caption("Sorted by wickets first, then fewer runs conceded.")
        bf = bw['best_figures'].copy()
        if not bf.empty:
            bf['figure'] = bf['wickets'].astype(int).astype(str) + '/' + bf['runs_given'].astype(int).astype(str)
        st.dataframe(
            bf[['bowler', 'figure', 'wickets', 'runs_given', 'match_id', 'inning']]
            .rename(columns={'match_id': 'Match', 'inning': 'Innings', 'runs_given': 'Runs Given'}),
            use_container_width=True,
            hide_index=True,
            height=420,
        )

    with tab4:
        st.subheader("Season Leaders")
        st.dataframe(bw['season_top'], use_container_width=True, hide_index=True, height=280)
        if not bw['season_top'].empty:
            st.pyplot(
                chart_record_barv(bw['season_top'], 'season', 'wickets', 'Top Bowler Wickets by Season', 'Wickets', len(bw['season_top']), '#f0824a'),
                use_container_width=True,
            )
        st.markdown("##### Highest Runs Conceded in a Single Over")
        st.dataframe(bw['over_runs'].head(15), use_container_width=True, hide_index=True, height=320)


# ── FIELDING RECORDS ─────────────────────────────────────────
elif page == "fielding_records":
    render_page_header("Fielding Records", "Catches, run-outs, and team-level catching output")
    fd = view_data['fielding']

    fm1, fm2, fm3 = st.columns(3)
    fm1.metric("Total Catches", f"{int(fd['catches']['catches'].sum()):,}" if not fd['catches'].empty else "0")
    fm2.metric("Total Run-Outs", f"{int(fd['runouts']['run_outs'].sum()):,}" if not fd['runouts'].empty else "0")
    fm3.metric("Teams with Catches", f"{len(fd['catches_team'])}")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Most Catches", "Most Run-Outs", "Team Catches"])
    with tab1:
        st.subheader("Top Fielders by Catches")
        st.pyplot(chart_record_barh(fd['catches'], 'catches', 'fielder', 'Most Catches', 'Catches', 20, '#3ec6a0'), use_container_width=True)
        st.dataframe(fd['catches'].head(20), use_container_width=True, hide_index=True, height=420)
    with tab2:
        st.subheader("Top Fielders by Run-Outs")
        st.pyplot(chart_record_barh(fd['runouts'], 'run_outs', 'fielder', 'Most Run-Outs', 'Run-Outs', 20, '#f0824a'), use_container_width=True)
        st.dataframe(fd['runouts'].head(20), use_container_width=True, hide_index=True, height=420)
    with tab3:
        st.subheader("Team Catching Output")
        c1, c2 = st.columns([1.3, 1])
        with c1:
            st.pyplot(chart_record_barh(fd['catches_team'], 'catches', 'bowling_team', 'Most Team Catches', 'Catches', 20, '#7c6af7'), use_container_width=True)
        with c2:
            st.dataframe(fd['catches_team'].head(20), use_container_width=True, hide_index=True, height=420)


# ── TEAM SCORES ──────────────────────────────────────────────
elif page == "team_scores":
    render_page_header("Team Scores", "Highest/lowest team totals and season top-scoring batters")
    ts = view_data['team_scores']

    tab1, tab2, tab3 = st.tabs(["Highest Totals", "Lowest Totals", "Season Top Batter"])
    with tab1:
        st.dataframe(ts['highest'], use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(ts['lowest'], use_container_width=True, hide_index=True)
    with tab3:
        st.dataframe(ts['season_top_batter'], use_container_width=True, hide_index=True)
        if not ts['season_top_batter'].empty:
            st.pyplot(
                chart_record_barv(ts['season_top_batter'], 'season', 'runs', 'Highest Runs per Season', 'Runs', len(ts['season_top_batter']), '#7c6af7'),
                use_container_width=True,
            )


# ── TEAM & CAPTAIN RECORDS ───────────────────────────────────
elif page == "team_captain_records":
    render_page_header("Team & Captain Records", "Team wins overview")
    mr = view_data['match_records']

    st.pyplot(chart_record_barh(mr['team_wins'], 'wins', 'team', 'Most Team Wins', 'Wins', 15, '#3ec6a0'), use_container_width=True)
    tbl = mr['team_wins'].copy()
    tbl['win_rate'] = tbl['win_rate'].map('{:.1%}'.format)
    st.dataframe(tbl, use_container_width=True, hide_index=True)
