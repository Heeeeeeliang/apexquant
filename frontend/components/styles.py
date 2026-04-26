import streamlit as st

LOGO_SVG = """
<svg width="32" height="32" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect width="48" height="48" rx="8" fill="#0d0d14"/>
  <polygon points="24,7 43,40 5,40" fill="none" stroke="#f5a623" stroke-width="2.2" stroke-linejoin="round"/>
  <polygon points="24,19 34,40 14,40" fill="#f5a623" fill-opacity="0.15" stroke="none"/>
  <circle cx="24" cy="18" r="2.8" fill="#f5a623"/>
</svg>
"""

def inject_global_css():
    st.html("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
    /* -- Reset & Base -- */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }
    .stApp {
        background-color: #0e1117;
        color: #e6edf3;
    }

    /* -- Metric cards -- */
    [data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 16px 20px;
    }
    [data-testid="stMetricLabel"] {
        color: #8b949e !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    [data-testid="stMetricValue"] {
        color: #e6edf3 !important;
        font-size: 24px !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    [data-testid="stMetricDelta"] svg { display: none; }
    [data-testid="stMetricDelta"] [data-testid="stMetricDeltaIcon-Up"] + div { color: #3fb950 !important; }
    [data-testid="stMetricDelta"] [data-testid="stMetricDeltaIcon-Down"] + div { color: #f85149 !important; }

    /* -- Buttons -- */
    .stButton > button {
        background-color: #21262d !important;
        color: #c9d1d9 !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 500 !important;
        font-size: 13px !important;
        padding: 6px 16px !important;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        background-color: #30363d !important;
        border-color: #8b949e !important;
        color: #e6edf3 !important;
    }
    .stButton > button[kind="primary"] {
        background-color: #f5a623 !important;
        color: #0d0d14 !important;
        border-color: #f5a623 !important;
        font-weight: 600 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #e09520 !important;
        border-color: #e09520 !important;
    }

    /* -- Inputs / Selects -- */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
        color: #e6edf3 !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 13px !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #f5a623 !important;
        box-shadow: 0 0 0 3px rgba(245,166,35,0.12) !important;
    }
    .stSelectbox > div > div {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
        color: #e6edf3 !important;
    }

    /* -- Dataframes / Tables -- */
    [data-testid="stDataFrame"] {
        border: 1px solid #21262d;
        border-radius: 8px;
        overflow: hidden;
    }
    [data-testid="stDataFrame"] th {
        background-color: #161b22 !important;
        color: #8b949e !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        border-bottom: 1px solid #21262d !important;
    }
    [data-testid="stDataFrame"] td {
        background-color: #0e1117 !important;
        color: #c9d1d9 !important;
        font-size: 13px !important;
        border-bottom: 1px solid #161b22 !important;
    }

    /* -- Tabs -- */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent !important;
        border-bottom: 1px solid #21262d !important;
        gap: 0;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        color: #8b949e !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        padding: 8px 16px !important;
        border-bottom: 2px solid transparent !important;
    }
    .stTabs [aria-selected="true"] {
        color: #f5a623 !important;
        border-bottom-color: #f5a623 !important;
    }

    /* -- Expander -- */
    .streamlit-expanderHeader {
        background-color: #161b22 !important;
        border: 1px solid #21262d !important;
        border-radius: 6px !important;
        color: #c9d1d9 !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 13px !important;
        font-weight: 500 !important;
    }

    /* -- Dividers -- */
    hr {
        border-color: #21262d !important;
        margin: 20px 0 !important;
    }

    /* -- Page headings -- */
    h1 { font-size: 22px !important; font-weight: 700 !important; letter-spacing: -0.02em; color: #e6edf3 !important; }
    h2 { font-size: 16px !important; font-weight: 600 !important; color: #c9d1d9 !important; }
    h3 { font-size: 13px !important; font-weight: 600 !important; color: #8b949e !important; text-transform: uppercase; letter-spacing: 0.08em; }

    /* -- Status badge helpers -- */
    .aq-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .aq-badge-green  { background: rgba(63,185,80,0.15);  color: #3fb950; border: 1px solid rgba(63,185,80,0.3);  }
    .aq-badge-red    { background: rgba(248,81,73,0.15);  color: #f85149; border: 1px solid rgba(248,81,73,0.3);  }
    .aq-badge-orange { background: rgba(245,166,35,0.15); color: #f5a623; border: 1px solid rgba(245,166,35,0.3); }
    .aq-badge-muted  { background: rgba(139,148,158,0.1); color: #8b949e; border: 1px solid rgba(139,148,158,0.2); }

    /* -- Hide Streamlit chrome -- */
    #MainMenu { visibility: hidden !important; }
    footer { visibility: hidden !important; }

    /* -- Force dark background on main container -- */
    .main .block-container {
        background-color: #0e1117 !important;
        padding-top: 24px !important;
        max-width: 100% !important;
    }

    /* -- Prevent white flash on inputs -- */
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div,
    div[data-baseweb="textarea"] > div {
        background-color: #161b22 !important;
        border-color: #30363d !important;
    }

    /* -- Plotly chart background fix -- */
    .js-plotly-plot .plotly .bg {
        fill: #161b22 !important;
    }

    /* -- Prevent white popover/dropdown flash -- */
    [data-baseweb="popover"] {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
    }
    ul[data-testid="stSelectboxVirtualDropdown"] {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
    }
    ul[data-testid="stSelectboxVirtualDropdown"] li:hover {
        background-color: #21262d !important;
    }
    </style>
    """)


def render_logo(show_text=True):
    """Render ApexQuant logo in sidebar header."""
    text_html = """
    <span style="
        font-family: 'Outfit', sans-serif;
        font-size: 17px;
        font-weight: 700;
        letter-spacing: -0.01em;
        color: #e6edf3;
        vertical-align: middle;
        margin-left: 10px;
    ">Apex<span style="color:#f5a623;">Quant</span></span>
    """ if show_text else ""

    st.html(f"""
    <div style="display:flex; align-items:center; padding: 20px 16px 16px 16px; border-bottom: 1px solid #21262d; margin-bottom: 8px;">
        {LOGO_SVG}
        {text_html}
    </div>
    """)
