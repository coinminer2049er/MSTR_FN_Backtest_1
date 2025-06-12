import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import json
import sys
from streamlit.components.v1 import html
import time
import qrcode
from PIL import Image # Pillow is needed by qrcode for image manipulation
import requests # For making HTTP requests to fetch live F&G index
import datetime # For handling date/time with F&G index

# Check Streamlit version and explicitly exit if too old
try:
    import streamlit
    st_version = streamlit.__version__
    # st.sidebar.info(f"Streamlit Version Detected: {st_version}") # REMOVED: Streamlit version display
    if st_version < "1.12.0":
        st.error(f"Streamlit version {st_version} detected. This app requires 1.12.0 or higher. Please upgrade using `pip install --upgrade streamlit`.")
        st.stop() # This should stop the app execution
except ImportError:
    st.error("Streamlit is not installed. Please install it using `pip install streamlit`.")
    st.stop()


# --- NEW APP TITLE AND SHORT DESCRIPTION ---
st.title("Backtest your MSTR buying strategy using real historical data")
st.markdown("Explore detailed strategy information, app instructions, and more below.")

# --- Function to fetch live F&G Index ---
@st.cache_data(ttl=3600) # Cache for 1 hour to avoid excessive API calls
def get_live_fg_index():
    try:
        # Alternative.me provides a JSON API for historical data, usually including the latest.
        # This endpoint is generally public and does not require an API key.
        response = requests.get("https://api.alternative.me/fng/?limit=1")
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        if data and data["data"]:
            latest_entry = data["data"][0]
            value = latest_entry["value"]
            sentiment = latest_entry["value_classification"]
            timestamp = int(latest_entry["timestamp"])
            # Changed date_str format to remove time and UTC
            date_str = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
            return f"{value} ({sentiment}) - Last updated: {date_str}"
    except requests.exceptions.RequestException as e:
        st.warning(f"Could not fetch live F&G Index: {e}. Displaying historical data only.")
    except (KeyError, IndexError) as e:
        st.warning(f"Unexpected data format from F&G API: {e}. Displaying historical data only.")
    return "N/A (Failed to load live data)"


# --- Detailed sections in Expanders (Reordered) ---

with st.expander("What is the MSTR F&G Strategy?"):
    st.markdown("""
    The **MSTR Bitcoin Fear & Greed Strategy** is a trading system that harnesses market sentiment to guide investments in MicroStrategy (**MSTR**) stock. It uses the **[Bitcoin Fear & Greed (F&G) Index)](https://alternative.me/crypto/fear-and-greed-index/)** to pinpoint trading opportunities: buying when fear is high (low F&G scores) and selling when greed peaks (high F&G scores). Optional **Bitcoin (BTC) price confirmation** aligns MSTR trades with BTC market trends, leveraging their correlation. Customizable settings, like initial capital, trade size, and cooldown periods, let users tailor the strategy to their preferences.

    For more information on MicroStrategy's Bitcoin strategy, visit the official **[Strategy Investor Relations](https://www.microstrategy.com/investor-relations)** page.
    """)

with st.expander("Why buy MSTR over Bitcoin?"):
    st.markdown("""
    Choosing to invest in MSTR (MicroStrategy) as opposed to directly buying Bitcoin
    is a strategy some investors adopt for various reasons:

    * **Tax-Advantaged Accounts:** MSTR can be purchased within tax-free savings accounts, such as an ISA (UK) or IRA (US), potentially allowing for tax-free profits. This can also help avoid the current ambiguous crypto tax guidelines and the risks associated with incorrect tax reporting.
    * **Traditional Market Accessibility:** MSTR is a NASDAQ-listed public company. This means it can be bought and sold through conventional stock brokerage accounts, making it familiar and accessible to investors who prefer not to use cryptocurrency exchanges or deal with direct crypto custody.
    * **Corporate Structure & Governance:** Investing in MSTR means you're investing in a legally established corporation. While highly concentrated in Bitcoin, the company has an existing software business, management team, and follows traditional financial reporting standards. For some, this corporate wrapper offers a perceived layer of familiarity or regulatory clarity compared to holding Bitcoin directly.
    * **Leveraged Bitcoin Exposure:** A significant portion of MicroStrategy's corporate treasury is allocated to Bitcoin. Historically, MicroStrategy has used debt financing to acquire large amounts of Bitcoin. This strategy can provide investors with a leveraged (though often more volatile) exposure to Bitcoin's price movements that might be difficult or costly to achieve through direct personal borrowing.
    * **Michael Saylor's Conviction:** MicroStrategy's Executive Chairman, Michael Saylor, is a vocal and highly influential advocate for Bitcoin. His unwavering commitment to a Bitcoin-centric corporate treasury strategy provides a unique, ideologically-driven leadership that resonates with certain investors.

    **Important Disclaimer:**
    Investing in MSTR carries unique risks beyond direct Bitcoin exposure. This includes:
    * **Stock Market Volatility:** MSTR's stock price is subject to equity market fluctuations and company-specific news, in addition to Bitcoin's volatility.
    * **Premium/Discount to NAV:** MSTR's stock can trade at a significant premium or discount relative to the value of its underlying Bitcoin holdings, influenced by market sentiment and speculation.
    * **Debt Risk:** The company's debt-financed Bitcoin acquisitions introduce additional financial risk.

    This information is for educational purposes only and should not be considered financial advice. Bitcoin and MSTR investments are highly volatile and speculative. Always conduct your own thorough due diligence and consult with a qualified financial advisor before making any investment decisions.
    """)

with st.expander("About this App"):
    st.markdown("""
    The **MSTR F&G Strategy Backtest App** empowers users to simulate and optimize this trading strategy with historical data. Its intuitive interface features **sliders to adjust parameters** like F&G thresholds, position sizes, and BTC confirmation, instantly showing their impact on returns, trades, and portfolio performance. Users can **save and load strategy configurations**, view detailed yearly metrics, and explore MSTR and BTC price charts with trade signals, which can be downloaded as images. Built for traders and investors, the app offers a **risk-free way to refine strategies and gain insights.**
    """)
    st.warning("""
    **Disclaimer on Usage:**
    All information, content, and tools provided within this application are for informational and educational purposes only. They are not intended as financial, investment, or trading advice. Past performance is not indicative of future results. The simulations and backtests presented are based on historical data and do not guarantee actual future returns. Market conditions can change rapidly.

    **By using this app, you acknowledge and agree that:**
    * You are solely responsible for any investment decisions you make.
    * Any losses incurred, directly or indirectly, from the use of the information or strategies presented in this app are your sole responsibility.
    * You should consult with a qualified financial professional before making any real investment decisions.
    """)

# --- Display Live F&G Index and MSTR Price Info after expanders ---
st.subheader("Current Market Insights")
col_live1, col_live2 = st.columns(2)

with col_live1:
    st.markdown(f"**Bitcoin Fear & Greed Index:**")
    st.info(get_live_fg_index())

with col_live2:
    st.markdown(f"**Latest MSTR Price:**")
    # Wrapped the button in st.warning to give it a yellowish background
    with st.warning(" "): # Added a space as body argument. This is a workaround for old versions
                          # if for some reason the st.stop() isn't working, but with 1.45.1
                          # it should just work with empty body.
        st.link_button("Check MSTR on Nasdaq", "https://www.nasdaq.com/market-activity/stocks/mstr")


# Load data
@st.cache_data
def load_data():
    try:
        # Load MSTR data
        mstr_data = pd.read_csv("mstr.csv") # Already lowercase
        mstr_data["Date"] = pd.to_datetime(mstr_data["Date"], errors="coerce")

        # Find adjusted close column
        adj_close_col = None
        for col in ["Adj Close", "Adjusted Close", "Close", "adjusted_close", "Close Adj"]:
            if col in mstr_data.columns:
                adj_close_col = col
                break
        if adj_close_col is None:
            st.error("No adjusted close column found in mstr.csv.")
            return None, None

        mstr_data = mstr_data[["Date", adj_close_col, "High", "Low"]].dropna()
        mstr_data = mstr_data.rename(columns={adj_close_col: "Adj Close"})

        # Load F&G data
        fg_data = pd.read_csv("fg_index.csv") # Already lowercase
        fg_data["Date"] = pd.to_datetime(fg_data["Date"], errors="coerce")
        fg_data = fg_data[["Date", "F&G", "BTC_Open", "BTC_Close", "BTC_High", "BTC_Low"]].dropna()

        # Merge data
        df = pd.merge(mstr_data, fg_data, on="Date", how="inner")
        df = df.sort_values("Date").reset_index(drop=True)

        if df.empty:
            st.error("No overlapping dates between mstr.csv and fg_index.csv.")
            return None, None

        # Calculate returns and correlation
        df["MSTR_Return"] = df["Adj Close"].pct_change()
        df["BTC_Return"] = df["BTC_Close"].pct_change()
        df["Correlation"] = df["MSTR_Return"].rolling(30, min_periods=30).corr(df["BTC_Return"])

        return df, mstr_data.columns.tolist()

    except FileNotFoundError as e:
        st.error(f"CSV file not found: {e}. Ensure mstr.csv and fg_index.csv are in the correct directory.")
        return None, None
    except Exception as e:
        st.error(f"Error loading CSV files: {e}")
        return None, None

df, mstr_columns = load_data()
if df is None:
    st.stop()

# Initialize session state
if "slider_values" not in st.session_state:
    st.session_state.slider_values = {
        "initial_capital": 40000,
        "buy_threshold": 25,
        "sell_threshold": 75,
        "buy_pct": 15.0,
        "sell_pct": 5.0,
        "btc_confirm": False,
        "cooldown_days": 0
    }

if "active_slider" not in st.session_state:
    st.session_state.active_slider = None

if "last_update_time" not in st.session_state:
    st.session_state.last_update_time = 0

if "backtest_cache" not in st.session_state:
    st.session_state.backtest_cache = {}

if "run_backtest" not in st.session_state:
    st.session_state.run_backtest = True

if "loaded_config" not in st.session_state:
    st.session_state.loaded_config = None

# Simplified JavaScript for slider focus (kept for best effort keyboard focus)
js_code = """
<script>
function handleArrowKeys(event) {
    if (['ArrowLeft', 'ArrowRight'].includes(event.key)) {
        const slider = event.target;
        const label = slider.getAttribute('aria-label');
        window.parent.postMessage({
            type: 'SET_ACTIVE_SLIDER',
            sliderId: label
        }, '*');
    }
}

document.addEventListener('DOMContentLoaded', function() {
    const setupSliders = function() {
        const sliders = document.querySelectorAll('input[data-testid="stSlider"]');
        sliders.forEach(function(slider) {
            slider.removeEventListener('keydown', handleArrowKeys);
            slider.removeEventListener('focus', handleFocus);

            slider.addEventListener('keydown', handleArrowKeys);
            slider.addEventListener('focus', handleFocus);
        });
    };

    function handleFocus(event) {
        const label = event.target.getAttribute('aria-label');
        window.parent.postMessage({
            type: 'SET_ACTIVE_SLIDER',
            sliderId: label
        }, '*');
    }

    setupSliders();

    const observer = new MutationObserver(function(mutations) {
        setupSliders();
    });

    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
});

window.addEventListener('message', function(event) {
    if (event.data.type === 'SET_ACTIVE_SLIDER') {
        const slider = document.querySelector(`input[data-testid="stSlider"][aria-label="${event.data.sliderId}"]`);
        if (slider) {
            slider.focus();
        }
    }
});
</script>
"""
html(js_code, height=0)

# Sidebar inputs
st.sidebar.header("Strategy Parameters")

# We will keep the debounce logic for mouse interactions,
# but it's important to understand it's the bottleneck for keyboard speed.
# For the new buttons, we will bypass it.
def update_slider_value(slider_key, value):
    st.session_state.slider_values[slider_key] = value
    st.session_state.active_slider = slider_key # Still useful for visual feedback
    st.session_state.run_backtest = True # Trigger recalculation

def on_slider_change(slider_key):
    # This function is called by Streamlit's slider on_change.
    # It still uses a debounce to prevent excessive re-runs during dragging.
    current_time = time.time()
    if current_time - st.session_state.last_update_time > 0.5:
        widget_value = st.session_state[slider_key]
        if slider_key in ["buy_pct", "sell_pct"]:
            widget_value = float(widget_value)
        elif slider_key == "btc_confirm":
            widget_value = bool(widget_value)
        else:
            widget_value = int(widget_value)
        update_slider_value(slider_key, widget_value)
        st.session_state.last_update_time = current_time

# Helper to create sliders with adjacent buttons
def create_slider_with_buttons(label, min_val, max_val, current_value_key, step, is_percentage=False):
    st.sidebar.write(label)
    cols = st.sidebar.columns([1, 6, 1]) # Adjust column ratios for button placement

    # Decrement button
    with cols[0]:
        if cols[0].button("◀️", key=f"btn_prev_{current_value_key}"):
            new_val = st.session_state.slider_values[current_value_key] - step
            if is_percentage:
                new_val = max(min_val, new_val)
            else:
                new_val = max(min_val, int(new_val)) # Ensure integer for non-percentage
            update_slider_value(current_value_key, new_val)
            st.rerun() # Rerun immediately to update slider position and trigger backtest

    # Slider
    with cols[1]:
        slider_display_value = st.session_state.slider_values[current_value_key]
        if is_percentage:
            slider_display_value = float(slider_display_value) # Ensure float
        else:
            slider_display_value = int(slider_display_value) # Ensure int

        slider_val = st.slider(
            " ", # Empty label as the main label is above
            min_val, max_val,
            slider_display_value,
            step=step,
            key=current_value_key,
            on_change=on_slider_change,
            args=(current_value_key,)
        )
        # Update session state from slider if it changes manually (mouse drag)
        # This check is important to sync the button updates with the slider display
        if slider_val != st.session_state.slider_values[current_value_key]:
            update_slider_value(current_value_key, slider_val)


    # Increment button
    with cols[2]:
        if cols[2].button("▶️", key=f"btn_next_{current_value_key}"):
            new_val = st.session_state.slider_values[current_value_key] + step
            if is_percentage:
                new_val = min(max_val, new_val)
            else:
                new_val = min(max_val, int(new_val)) # Ensure integer for non-percentage
            update_slider_value(current_value_key, new_val)
            st.rerun() # Rerun immediately to update slider position and trigger backtest

# Define sliders using the helper function
create_slider_with_buttons(
    "Starting Capital ($)",
    0, 100000,
    "initial_capital",
    250
)
initial_capital = st.session_state.slider_values["initial_capital"]


create_slider_with_buttons(
    "F&G Buy Threshold (≤)",
    0, 100,
    "buy_threshold",
    1
)
buy_threshold = st.session_state.slider_values["buy_threshold"]


create_slider_with_buttons(
    "F&G Sell Threshold (≥)",
    0, 100,
    "sell_threshold",
    1
)
sell_threshold = st.session_state.slider_values["sell_threshold"]


create_slider_with_buttons(
    "Buy Percentage (%)",
    0.0, 50.0,
    "buy_pct",
    0.5,
    is_percentage=True
)
buy_pct = st.session_state.slider_values["buy_pct"] / 100 # Convert to fraction for backtest


create_slider_with_buttons(
    "Sell Percentage (%)",
    0.0, 50.0,
    "sell_pct",
    0.5,
    is_percentage=True
)
sell_pct = st.session_state.slider_values["sell_pct"] / 100 # Convert to fraction for backtest


# BTC Confirmation Checkbox (no buttons needed)
# Using a direct update here as it's a simple toggle and doesn't benefit from the debouncer
btc_confirm = st.sidebar.checkbox(
    "Use BTC Confirmation (Buy: BTC < -5%, Sell: BTC > 5%)",
    st.session_state.slider_values["btc_confirm"],
    key="btc_confirm_checkbox" # Changed key to avoid conflict if 'btc_confirm' was used elsewhere
)
# Check if the checkbox state has changed and update session_state
if btc_confirm != st.session_state.slider_values["btc_confirm"]:
    update_slider_value("btc_confirm", btc_confirm)
    st.rerun() # Trigger a rerun to apply the change immediately


create_slider_with_buttons(
    "Cooldown Days",
    0, 10,
    "cooldown_days",
    1
)
cooldown_days = st.session_state.slider_values["cooldown_days"]

# Validate thresholds
if buy_threshold >= sell_threshold:
    st.sidebar.warning("Buy Threshold should be less than Sell Threshold for meaningful trades.")

# Save/Load configuration
st.sidebar.subheader("Save/Load Strategy")
config = {
    "initial_capital": initial_capital,
    "buy_threshold": buy_threshold,
    "sell_threshold": sell_threshold,
    "buy_pct": buy_pct * 100, # Save as percentage
    "sell_pct": sell_pct * 100, # Save as percentage
    "btc_confirm": btc_confirm,
    "cooldown_days": cooldown_days
}

config_json = json.dumps(config, indent=2)
st.sidebar.download_button(
    label="Save Strategy Configuration",
    data=config_json,
    file_name="strategy_config.json",
    mime="application/json"
)

# Load configuration
uploaded_config = st.sidebar.file_uploader("Load Strategy Configuration", type=["json"])
if uploaded_config is not None:
    try:
        loaded_config = json.load(uploaded_config)
        required_keys = ["initial_capital", "buy_threshold", "sell_threshold",
                         "buy_pct", "sell_pct", "btc_confirm", "cooldown_days"]

        if not all(key in loaded_config for key in required_keys):
            st.sidebar.error("Invalid configuration file: Missing required parameters.")
        else:
            # Validate and clamp loaded values
            loaded_values = {
                "initial_capital": max(0, min(100000, int(loaded_config["initial_capital"]))),
                "buy_threshold": max(0, min(100, int(loaded_config["buy_threshold"]))),
                "sell_threshold": max(0, min(100, int(loaded_config["sell_threshold"]))),
                "buy_pct": max(0.0, min(50.0, float(loaded_config["buy_pct"]))),
                "sell_pct": max(0.0, min(50.0, float(loaded_config["sell_pct"]))),
                "btc_confirm": bool(loaded_config["btc_confirm"]),
                "cooldown_days": max(0, min(10, int(loaded_config["cooldown_days"])))
            }

            st.session_state.loaded_config = loaded_values
            st.sidebar.success("Configuration loaded successfully. Click 'Apply Loaded Config' to update.")

    except json.JSONDecodeError as e:
        st.sidebar.error(f"Invalid JSON file: {str(e)}")
    except Exception as e:
        st.sidebar.error(f"Error loading configuration: {str(e)}")

# Apply loaded configuration
if st.sidebar.button("Apply Loaded Config") and st.session_state.loaded_config is not None:
    st.session_state.slider_values = st.session_state.loaded_config.copy()
    st.session_state.run_backtest = True
    st.session_state.active_slider = None
    st.session_state.loaded_config = None
    st.sidebar.success("Applied loaded configuration.")
    st.rerun() # Rerun to apply changes immediately

# Backtest function with optimizations
@st.cache_data(show_spinner=False) # Use st.cache_data for backtest results
def backtest_strategy(df, buy_threshold, sell_threshold, buy_pct, sell_pct, btc_confirm, cooldown_days, initial_capital):
    # This cache key is now just for the function, not session state
    # No need for manual cache in session state if using @st.cache_data
    # Cache key generated automatically by Streamlit

    cash = initial_capital
    shares = 0
    costs = 0
    trades = []
    last_trade_date = None
    skipped_trades = 0
    portfolio_values = []

    progress_bar = st.progress(0)
    total_rows = len(df)

    for i, row in df.iterrows():
        progress_bar.progress(min((i + 1) / total_rows, 1.0))

        # Check cooldown period
        if last_trade_date and (row["Date"] - last_trade_date).days < cooldown_days:
            continue

        btc_condition_buy = not btc_confirm or (btc_confirm and row["BTC_Return"] < -0.05)
        btc_condition_sell = not btc_confirm or (btc_confirm and row["BTC_Return"] > 0.05)

        # Buy condition
        if row["F&G"] <= buy_threshold and btc_condition_buy:
            buy_amt = cash * buy_pct
            total_cost = buy_amt + 15 + buy_amt * 0.002 # $15 fixed fee + 0.2% commission

            if total_cost <= cash:
                new_shares = buy_amt / row["Adj Close"]
                cash -= total_cost
                shares += new_shares
                costs += 15 + buy_amt * 0.002

                trades.append({
                    "Date": row["Date"],
                    "Type": "Buy",
                    "Price": row["Adj Close"],
                    "Shares": new_shares,
                    "Size": new_shares * row["Adj Close"],
                    "BTC_Price": row["BTC_Close"],
                    "F&G": row["F&G"]
                })
                last_trade_date = row["Date"]
            else:
                skipped_trades += 1

        # Sell condition
        elif row["F&G"] >= sell_threshold and shares > 0 and btc_condition_sell:
            sell_shares = shares * sell_pct
            proceeds = sell_shares * row["Adj Close"]
            cash += proceeds - 15 - proceeds * 0.002 # $15 fixed fee + 0.2% commission
            shares -= sell_shares
            costs += 15 + proceeds * 0.002

            trades.append({
                "Date": row["Date"],
                "Type": "Sell",
                "Price": row["Adj Close"],
                "Shares": sell_shares,
                "Size": sell_shares * row["Adj Close"],
                "BTC_Price": row["BTC_Close"],
                "F&G": row["F&G"]
            })
            last_trade_date = row["Date"]


        # Record portfolio value
        portfolio_value = cash + shares * row["Adj Close"]
        portfolio_values.append({
            "Date": row["Date"],
            "Cash": cash,
            "Shares": shares,
            "Portfolio_Value": portfolio_value,
            "Costs": costs
        })

    progress_bar.empty()

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    portfolio_df = pd.DataFrame(portfolio_values) if portfolio_values else pd.DataFrame()

    final_value = portfolio_df["Portfolio_Value"].iloc[-1] if not portfolio_df.empty else initial_capital
    profit = final_value - initial_capital

    years = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days / 365.25 if len(df) > 1 else 1
    annualized_return = ((final_value / initial_capital) ** (1 / years) - 1) * 100 if years > 0 and initial_capital > 0 else 0

    result = {
        "Profit": profit,
        "Final Shares": shares,
        "Final Cash": cash,
        "Costs": costs,
        "Annualized Return": annualized_return,  # Already percentage from calculation above
        "Trades": len(trades_df),
        "Trades_df": trades_df,
        "Skipped_Trades": skipped_trades,
        "Portfolio_df": portfolio_df
    }

    return result

# Run backtest
try:
    with st.spinner("Running backtest..."):
        result = backtest_strategy(df, buy_threshold, sell_threshold, buy_pct, sell_pct, btc_confirm, cooldown_days, initial_capital)
except Exception as e:
    st.error(f"Error running backtest: {e}")
    st.stop()


# Yearly breakdown function
def compute_yearly_metrics(df, portfolio_df, trades_df):
    if portfolio_df.empty or df.empty:
        return pd.DataFrame()

    portfolio_df["Year"] = portfolio_df["Date"].dt.year
    trades_df["Year"] = trades_df["Date"].dt.year if not trades_df.empty else pd.Series()

    yearly_metrics = []
    for year in sorted(portfolio_df["Year"].unique()):
        year_portfolio = portfolio_df[portfolio_df["Year"] == year]
        year_trades = trades_df[trades_df["Year"] == year] if not trades_df.empty else pd.DataFrame()

        if year_portfolio.empty:
            continue

        # Get start/end values for the year
        # Use .copy() to avoid SettingWithCopyWarning if you modify these later
        start_value = year_portfolio["Portfolio_Value"].iloc[0]
        end_value = year_portfolio["Portfolio_Value"].iloc[-1]
        start_costs = year_portfolio["Costs"].iloc[0]
        end_costs = year_portfolio["Costs"].iloc[-1]

        year_profit = end_value - start_value
        year_return = ((end_value / start_value) - 1) * 100 if start_value > 0 else 0
        year_costs = end_costs - start_costs
        final_shares = year_portfolio["Shares"].iloc[-1]
        final_cash = year_portfolio["Cash"].iloc[-1]
        num_trades = len(year_trades)

        yearly_metrics.append({
            "Year": year,
            "Profit": year_profit,
            "Annual Return (%)": year_return,
            "Final Cash": final_cash,
            "Total Costs": year_costs,
            "Number of Trades": num_trades
        })

    return pd.DataFrame(yearly_metrics)

# Compute yearly metrics
try:
    yearly_metrics_df = compute_yearly_metrics(df, result["Portfolio_df"], result["Trades_df"])
except Exception as e:
    st.error(f"Error computing yearly metrics: {e}")
    yearly_metrics_df = pd.DataFrame()

# Display yearly breakdown
st.header("Yearly Performance Breakdown")
if not yearly_metrics_df.empty:
    total_profit = yearly_metrics_df["Profit"].sum()

    # Define a function to apply color based on profit value
    def highlight_profit_cell(val):
        color = "white" # Default for zero
        if val > 0:
            color = "green"
        elif val < 0:
            color = "red"
        return f"color: {color};"

    # Create a Styler object from the original yearly_metrics_df
    # Apply the color function to the 'Profit' column using map
    styled_yearly_metrics_df = yearly_metrics_df.style.map(
        highlight_profit_cell,
        subset=['Profit'] # Apply only to the 'Profit' column
    )

    # Apply number formatting to all relevant columns within the Styler
    styled_yearly_metrics_df = styled_yearly_metrics_df.format({
        "Profit": "${:,.2f}",
        "Annual Return (%)": "{:.2f}%",
        "Final Cash": "${:,.2f}",
        "Total Costs": "${:,.2f}"
    })


    st.write("**Yearly Metrics**")
    st.dataframe(styled_yearly_metrics_df, hide_index=True)
    st.write(f"**Total Profit Across All Years**: ${total_profit:,.2f}")
else:
    st.warning("No yearly metrics available. Check data span or trades.")


# Display overall results
st.header("Overall Backtest Results")
col1, col2 = st.columns(2)

with col1:
    st.metric("Starting Capital", f"${initial_capital:,.2f}")

    profit_value = result['Profit']
    if profit_value >= 0:
        profit_color = "green"
        profit_sign = "+"
    else:
        profit_color = "red"
        profit_sign = "" # No sign for negative values
    st.markdown(
        f"""
        <div style="font-size: 14px; color: rgba(250, 250, 250, 0.6);">Profit</div>
        <div style="font-size: 36px; font-weight: 600; color: {profit_color};">
            {profit_sign}${profit_value:,.2f}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.metric("Annualized Return", f"{result['Annualized Return']:.2f}%")

with col2:
    st.metric("Final Shares", f"{result['Final Shares']:,.2f}")
    st.metric("Final Cash", f"${result['Final Cash']:,.2f}") # Corrected: used 'Final Cash' from result dict
    st.metric("Total Costs", f"${result['Costs']:,.2f}")

st.metric("Number of Trades", result["Trades"])
if result["Skipped_Trades"] > 0:
    st.warning(f"**Skipped Trades**: {result['Skipped_Trades']} due to insufficient funds.")


# AI Strategy Feedback
st.header("🤖 AI Strategy Feedback")
st.info(
    "**Disclaimer:** This feedback is generated by a rule-based AI simulation "
    "based on the backtest results and input parameters. It is intended for "
    "informational and exploratory purposes only and **does not constitute "
    "financial advice or a guarantee of future performance.** Always conduct "
    "your own research."
)

# New function for car comparison - now returns just the core sentence
def get_car_comparison_text(profit_amount):
    if profit_amount <= -5000: # Significant loss
        return f"${profit_amount:,.2f} 😭 Ouch, how are you going to feed the kids?!"
    elif profit_amount < 0: # Small to moderate loss
        return f"${profit_amount:,.2f} 😥 Oh no, I can see a baked beans on toast diet in your future."
    elif profit_amount < 100: # Very small profit
        return f"${profit_amount:,.2f} 🤔 You might afford a decent coffee."
    elif profit_amount < 1000: # Small profit, mountain bike range
        return f"${profit_amount:,.2f} Not bad, you might be able to get a slightly used mountain bike off Marketplace!"
    elif profit_amount < 5000:
        return f"${profit_amount:,.2f} Amazing! That's a solid down payment on a reliable used car."
    elif profit_amount < 15000:
        return f"${profit_amount:,.2f} Incredible! That's used Toyota Corolla money!"
    elif profit_amount < 30000:
        return f"${profit_amount:,.2f} Fantastic! That's enough for a nice, low-mileage used SUV."
    elif profit_amount < 50000:
        return f"${profit_amount:,.2f} Wow! You're looking at brand new Toyota Camry money!"
    elif profit_amount < 80000:
        return f"${profit_amount:,.2f} Splendid! That's entry-level luxury sedan (e.g., BMW 3-Series) money!"
    elif profit_amount < 150000:
        return f"${profit_amount:,.2f} Impressive! You're in Porsche Boxster/Cayman territory!"
    elif profit_amount < 250000:
        return f"${profit_amount:,.2f} Absolutely Stunning! That's Porsche 911 Targa money!"
    elif profit_amount < 500000:
        return f"${profit_amount:,.2f} Phenomenal! You're in exotic supercar territory – think Ferrari F8!"
    else: # profit_amount >= 500000
        return f"${profit_amount:,.2f} Astounding! That's multiple dream cars money! Or you can always cash out, buy Bitcoin and spend your retirement living in the Bitcoin Citadel alongside Michael Saylor!"


def generate_ai_feedback(results, params, yearly_metrics_df):
    feedback = []

    profit = results["Profit"]
    annual_return = results["Annualized Return"]
    num_trades = results["Trades"]
    total_costs = results["Costs"]
    skipped_trades = results["Skipped_Trades"]
    initial_capital = params["initial_capital"]
    buy_threshold = params["buy_threshold"]
    sell_threshold = params["sell_threshold"]
    # These parameters already hold the percentage value (e.g., 15.0), not a fraction.
    buy_pct_display = params["buy_pct"]
    sell_pct_display = params["sell_pct"]
    btc_confirm = params["btc_confirm"]
    cooldown_days = params["cooldown_days"]

    # Car Comparison Feedback
    feedback.append("---") # Markdown horizontal rule
    feedback.append(f"🚗 {get_car_comparison_text(profit)}")
    feedback.append("---") # Markdown horizontal rule
    feedback.append("") # Add an empty string for a paragraph break

    # Overall Performance Summary
    if annual_return > 15:
        feedback.append("✨ **Outstanding Performance!** Your strategy generated a strong annualized return, indicating excellent historical profitability.")
    elif annual_return > 5:
        feedback.append("📈 **Solid Performance.** The strategy delivered a positive annualized return, showing good potential.")
    elif annual_return >= 0:
        feedback.append("⚖️ **Modest Performance.** The strategy broke even or yielded a small positive return. There might be room for optimization.")
    else:
        feedback.append("📉 **Underperforming Strategy.** Your strategy resulted in a negative annualized return. It's highly recommended to revise the parameters.")
    feedback.append("") # Paragraph break

    # Profitability Analysis
    if profit > 0:
        # Added spaces around the bolded number for better markdown parsing
        feedback.append(f"The strategy achieved a net profit of **${profit:,.2f}** over the backtesting period.")
    else:
        # Added spaces around the bolded number for better markdown parsing
        feedback.append(f"The strategy incurred a net loss of **${abs(profit):,.2f}** over the backtesting period.")
    feedback.append("") # Paragraph break

    # Trading Activity & Efficiency
    if num_trades > 50:
        feedback.append(f"With **{num_trades}** trades executed, the strategy was quite active, potentially capturing many market movements.")
    elif num_trades > 10:
        feedback.append(f"A total of **{num_trades}** trades were executed, which is a moderate activity level.")
    else:
        feedback.append(f"Only **{num_trades}** trades were executed. This suggests a very selective strategy, or perhaps missed opportunities due to strict parameters.")

    if skipped_trades > 0 and num_trades == 0:
        feedback.append(f"⚠️ **Warning:** **{skipped_trades}** trades were skipped, and no trades were executed at all. This often means the initial capital (**${initial_capital:,.2f}**) or buy percentage (**{buy_pct_display:.1f}%**) was too low to cover transaction costs or minimum buy amounts for MSTR at historical prices.")
    elif skipped_trades > 0:
        feedback.append(f"Consider adjusting your capital or trade size: **{skipped_trades}** trades were skipped due to insufficient funds, which could have impacted overall performance.")
    feedback.append("") # Paragraph break

    # Explicitly formatting total_costs with dollar sign and commas for all cases
    if total_costs > (profit * 0.1) and profit > 0: # If costs are more than 10% of profit
        feedback.append(f"Transaction costs (**${total_costs:,.2f}**) seem somewhat high relative to the profit. Optimizing trade frequency or size might help reduce this impact.")
    elif profit <= 0 and total_costs > 0:
        feedback.append(f"Transaction costs (**${total_costs:,.2f}**) contributed to the loss. High costs can be detrimental when profitability is low.")
    elif total_costs == 0:
        feedback.append("Transaction costs were minimal or zero, indicating efficient trading or very few trades.")
    else: # Total costs are positive but not a significant percentage of profit (if profit is positive)
         feedback.append(f"Transaction costs (**${total_costs:,.2f}**) were managed well, not significantly impacting profitability.")
    feedback.append("") # Paragraph break

    # Parameter Insights
    feedback.append("**Insights on Your Parameters:**")
    if buy_threshold <= 20:
        feedback.append(f"- Your Buy Threshold (**{buy_threshold}**) is very conservative, aiming to buy only when Fear is extreme. This might lead to fewer trades but potentially higher conviction entry points.")
    elif buy_threshold <= 40:
        feedback.append(f"- Your Buy Threshold (**{buy_threshold}**) is moderately conservative. It seeks buying opportunities during significant market fear.")
    else:
        feedback.append(f"- Your Buy Threshold (**{buy_threshold}**) is relatively high. This may lead to more frequent buys but potentially less favorable entry points during periods of moderate fear.")

    if sell_threshold >= 80:
        feedback.append(f"- Your Sell Threshold (**{sell_threshold}**) is very aggressive, aiming to sell only when Greed is extreme. This seeks to maximize gains but risks missing exits if the market turns before extreme greed is reached.")
    elif sell_threshold >= 60:
        feedback.append(f"- Your Sell Threshold (**{sell_threshold}**) is moderately aggressive, aiming to sell when greed is elevated. This is a balanced approach to profit-taking.")
    else:
        feedback.append(f"- Your Sell Threshold (**{sell_threshold}**) is relatively low. This might lead to more frequent sells, potentially missing further upside during extended periods of greed.")

    if buy_pct_display <= 10.0:
        feedback.append(f"- Your Buy Percentage (**{buy_pct_display:.1f}%**) is quite small. This reduces risk but might lead to slower capital deployment and less significant gains when opportunities arise.")
    elif buy_pct_display >= 30.0:
        feedback.append(f"- Your Buy Percentage (**{buy_pct_display:.1f}%**) is relatively large. This can amplify gains but also increases risk per trade.")

    if sell_pct_display <= 10.0:
        feedback.append(f"- Your Sell Percentage (**{sell_pct_display:.1f}%**) is modest. This allows you to retain a portion of your holdings, potentially benefiting from further upside.")
    elif sell_pct_display >= 30.0:
        feedback.append(f"- Your Sell Percentage (**{sell_pct_display:.1f}%**) is substantial. This allows for quick profit realization but might lead to selling off too much too soon.")

    if btc_confirm:
        feedback.append("- **BTC Confirmation** is enabled, adding an extra layer of market trend validation. This typically reduces trade frequency but aims to increase conviction.")
    else:
        feedback.append("- **BTC Confirmation** is disabled. This means trades are solely based on the F&G index, potentially leading to more signals but without external market validation.")

    if cooldown_days > 0:
        feedback.append(f"- A **Cooldown Period of {cooldown_days}** days is applied. This prevents rapid successive trades, which can reduce transaction costs and mitigate overtrading, but might miss immediate subsequent opportunities.")
    else:
        feedback.append("- **No Cooldown Period** is applied, allowing for immediate re-entry/exit if conditions are met. Be mindful of potential overtrading and accumulating transaction costs.")
    feedback.append("") # Paragraph break

    # Further Suggestions
    feedback.append("**Suggestions for Further Exploration:**")
    feedback.append("- **Adjust parameters incrementally:** Try modifying one parameter at a time to understand its isolated impact on performance.")
    feedback.append("- **Explore different market cycles:** Consider how this strategy might perform in different bullish, bearish, or sideways market conditions (though data is fixed here).")
    feedback.append("- **Risk Management:** Always define your risk tolerance and position sizing before applying any strategy in live trading.")
    feedback.append("- **Diversification:** Consider how this MSTR-specific strategy fits into a broader diversified investment portfolio.")

    return "\n".join(feedback)

# Parameters for generating feedback
feedback_params = {
    "initial_capital": initial_capital,
    "buy_threshold": buy_threshold,
    "sell_threshold": sell_threshold,
    "buy_pct": st.session_state.slider_values["buy_pct"], # Pass actual percentage value
    "sell_pct": st.session_state.slider_values["sell_pct"], # Pass actual percentage value
    "btc_confirm": btc_confirm,
    "cooldown_days": cooldown_days
}

# GENERATE AND DISPLAY AI FEEDBACK HERE
ai_feedback_text = generate_ai_feedback(result, feedback_params, yearly_metrics_df)
st.markdown(ai_feedback_text)


# Display trade summary
st.subheader("Trade Summary")
trades_df = result["Trades_df"]
if not trades_df.empty:
    # Format trade data for display
    formatted_trades = trades_df.copy()
    formatted_trades["Price"] = formatted_trades["Price"].apply(lambda x: f"${x:,.2f}")
    formatted_trades["Size"] = formatted_trades["Size"].apply(lambda x: f"${x:,.2f}")
    formatted_trades["BTC_Price"] = formatted_trades["BTC_Price"].apply(lambda x: f"${x:,.2f}")

    st.dataframe(formatted_trades[["Date", "Type", "Price", "Shares", "Size", "BTC_Price", "F&G"]])

    # Prepare CSV download
    csv_buffer = io.StringIO()

    # Write strategy parameters
    csv_buffer.write("# Strategy Parameters\n")
    csv_buffer.write(f"Starting Capital,${initial_capital:,.2f}\n")
    csv_buffer.write(f"F&G Buy Threshold,{buy_threshold}\n")
    csv_buffer.write(f"F&G Sell Threshold,{sell_threshold}\n")
    csv_buffer.write(f"Buy Percentage,{buy_pct*100:.1f}%\n")
    csv_buffer.write(f"Sell Percentage,{sell_pct*100:.1f}%\n")
    csv_buffer.write(f"Use BTC Confirmation,{btc_confirm}\n")
    csv_buffer.write(f"Cooldown Days,{cooldown_days}\n")
    csv_buffer.write(f"Profit,${result['Profit']:,.2f}\n")
    csv_buffer.write(f"Skipped Trades,{result['Skipped_Trades']}\n\n")

    # Write yearly metrics if available
    if not yearly_metrics_df.empty:
        # For CSV, use the original numeric yearly_metrics_df, not the styled one.
        # This ensures the downloaded CSV has clean numbers.
        csv_buffer.write("# Yearly Performance\n")
        yearly_metrics_df.to_csv(csv_buffer, index=False)
        csv_buffer.write("\n")

    # Write trades
    csv_buffer.write("# Trade History\n")
    trades_df.to_csv(csv_buffer, index=False)

    csv_data = csv_buffer.getvalue()
    st.download_button(
        label="Download Trade Summary",
        data=csv_data,
        file_name="mstr_trades.csv",
        mime="text/csv"
    )
    csv_buffer.close()
else:
    st.write("No trades executed.")

# Plotting functions
def plot_price_path(price_col, title, line_color, trades_df, df, initial_capital, result):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df["Date"], df[price_col], label=title, color=line_color, alpha=0.7)

    if not trades_df.empty:
        buy_signals = trades_df[trades_df["Type"] == "Buy"]
        sell_signals = trades_df[trades_df["Type"] == "Sell"]

        # Ensure that 'Price' in trades_df aligns with the current price_col in df
        # We need to map the trade date to the actual price on that date from the main df
        # This is more robust than using trades_df['Price'] if trade price is different from daily close
        buy_prices = df.loc[df["Date"].isin(buy_signals["Date"]), price_col]
        sell_prices = df.loc[df["Date"].isin(sell_signals["Date"]), price_col]

        if not buy_signals.empty:
            ax.scatter(buy_signals["Date"], buy_prices,
                       color="green", label="Buy Signal", marker="^", s=100, zorder=5)

        if not sell_signals.empty:
            ax.scatter(sell_signals["Date"], sell_prices,
                       color="red", label="Sell Signal", marker="v", s=100, zorder=5)

    # Add strategy info - using parameters directly passed
    text_str = (
        f"Initial Capital: ${initial_capital:,.2f}\n"
        f"F&G Buy Threshold: {st.session_state.slider_values['buy_threshold']}\n"
        f"F&G Sell Threshold: {st.session_state.slider_values['sell_threshold']}\n"
        f"Buy Percentage: {st.session_state.slider_values['buy_pct']:.1f}%\n"
        f"Sell Percentage: {st.session_state.slider_values['sell_pct']:.1f}%\n"
        f"BTC Confirmation: {st.session_state.slider_values['btc_confirm']}\n"
        f"Cooldown Days: {st.session_state.slider_values['cooldown_days']}\n"
        f"Total Profit: ${result['Profit']:,.2f}"
    )

    ax.text(0.02, 0.98, text_str, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))

    ax.set_xlabel("Date")
    ax.set_ylabel(f"{title} Price (USD)")
    ax.set_title(f"{title} with Trades")
    ax.legend(loc="upper right")
    ax.grid(True)
    fig.autofmt_xdate() # Auto-format dates on x-axis

    return fig

# Plot MSTR price path
st.subheader("MSTR Price Path")
if not df.empty:
    fig_mstr = plot_price_path("Adj Close", "MSTR", "blue", result["Trades_df"], df, initial_capital, result)
    st.pyplot(fig_mstr)

    # Download button for the plot
    buf = io.BytesIO()
    fig_mstr.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        label="Download MSTR Graph",
        data=buf,
        file_name="mstr_price_path.png",
        mime="image/png"
    )
    buf.close()
    plt.close(fig_mstr) # Close the figure after displaying and saving
else:
    st.warning("No MSTR price data to plot.")

# Plot BTC price path
st.subheader("BTC Price Path")
if not df.empty:
    fig_btc = plot_price_path("BTC_Close", "BTC", "orange", result["Trades_df"], df, initial_capital, result)
    st.pyplot(fig_btc)

    # Download button for the plot
    buf = io.BytesIO()
    fig_btc.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        label="Download BTC Graph",
        data=buf,
        file_name="btc_price_path.png",
        mime="image/png"
    )
    buf.close()
    plt.close(fig_btc) # Close the figure after displaying and saving
else:
    st.warning("No BTC price data to plot.")

# Plot Portfolio Value Over Time
st.subheader("Portfolio Value Over Time")
if not result["Portfolio_df"].empty:
    fig_portfolio, ax_portfolio = plt.subplots(figsize=(14, 7))
    ax_portfolio.plot(result["Portfolio_df"]["Date"], result["Portfolio_df"]["Portfolio_Value"], label="Portfolio Value", color="purple")
    ax_portfolio.set_xlabel("Date")
    ax_portfolio.set_ylabel("Portfolio Value ($)")
    ax_portfolio.set_title("Portfolio Value Over Time")
    ax_portfolio.grid(True)
    ax_portfolio.legend()
    fig_portfolio.autofmt_xdate()

    st.pyplot(fig_portfolio)

    # Download button for the plot
    buf = io.BytesIO()
    fig_portfolio.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        label="Download Portfolio Value Graph",
        data=buf,
        file_name="portfolio_value_path.png",
        mime="image/png"
    )
    buf.close()
    plt.close(fig_portfolio) # Close the figure after displaying and saving
else:
    st.warning("No portfolio value data to plot.")


# --- Monetization/Support Section ---
st.sidebar.markdown("---")
st.sidebar.subheader("Support My Work!")
st.sidebar.markdown(
    "If you find this app helpful, please consider supporting its ongoing development. "
    "Your generosity is greatly appreciated!"
)

# st.sidebar.link_button("☕ Buy Me a Coffee", "https://ko-fi.com/yourusername_or_buymeacoffee_link") # REMOVED AS REQUESTED

# Your Nested SegWit Address (starts with '3')
YOUR_BITCOIN_ADDRESS = "39wDMRzG1u2n5BQsaF8LswaRwWLtnJRnkY"

st.sidebar.markdown("---")
st.sidebar.subheader("Donate with Bitcoin (BTC)")
st.sidebar.code(YOUR_BITCOIN_ADDRESS)

# Generate and display QR code on the fly
try:
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=5,
        border=4,
    )
    qr.add_data(f"bitcoin:{YOUR_BITCOIN_ADDRESS}")
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    st.sidebar.image(buf, caption="Scan to donate BTC")

except ImportError:
    st.sidebar.warning("Install 'qrcode' and 'Pillow' for QR code display: `pip install qrcode Pillow`")
except Exception as e:
    st.sidebar.error(f"Error generating QR code: {e}")


# --- Authorship/Source Code Section (Updated) ---
st.sidebar.markdown("---")
st.sidebar.info(
    "App developed by **miner2049er**. "
    "For more projects and insights, follow me on X: "
    "[**@coinminer2049er**](https://x.com/coinminer2049er)"
)
st.sidebar.markdown(
    "Find the source code on [GitHub](https://github.com/coinminer2049er/MSTR_FN_Backtest_1)." # <<< UPDATED LINK HERE
)
