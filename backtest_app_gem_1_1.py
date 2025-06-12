import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import json
import sys
from streamlit.components.v1 import html
import time

# Check Streamlit version
try:
    import streamlit
    st_version = streamlit.__version__
    if st_version < "1.12.0":
        st.error(f"Streamlit version {st_version} detected. Please upgrade to 1.12.0 or higher using `pip install --upgrade streamlit`.")
        st.stop()
except ImportError:
    st.error("Streamlit is not installed. Please install it using `pip install streamlit`.")
    st.stop()

# --- NEW APP TITLE AND SHORT DESCRIPTION ---
st.title("Backtest your MSTR buying strategy using real historical data")
st.markdown("Explore detailed strategy information, app instructions, and more below.")


# --- Detailed sections in Expanders (Reordered) ---

with st.expander("What is the MSTR F&G Strategy?"):
    st.markdown("""
    The **MSTR Bitcoin Fear & Greed Strategy** is a trading system that harnesses market sentiment to guide investments in Strategy (**MSTR**) stock. It uses the **Bitcoin Fear & Greed (F&G) Index** to pinpoint trading opportunities: buying when fear is high (low F&G scores) and selling when greed peaks (high F&G scores). Optional **Bitcoin (BTC) price confirmation** aligns MSTR trades with BTC market trends, leveraging their correlation. Customizable settings, like initial capital, trade size, and cooldown periods, let users tailor the strategy to their preferences.
    """)

with st.expander("Why buy MSTR over Bitcoin?"):
    st.markdown("""
    Choosing to invest in MSTR (Strategy) as opposed to directly buying Bitcoin
    is a strategy some investors adopt for various reasons:

    * **Tax-Advantaged Accounts:** MSTR can be purchased within tax-free savings accounts, such as an ISA (UK) or IRA (US), potentially allowing for tax-free profits. This can also help avoid the current ambiguous crypto tax guidelines and the risks associated with incorrect tax reporting.
    * **Traditional Market Accessibility:** MSTR is a NASDAQ-listed public company. This means it can be bought and sold through conventional stock brokerage accounts, making it familiar and accessible to investors who prefer not to use cryptocurrency exchanges or deal with direct crypto custody.
    * **Corporate Structure & Governance:** Investing in MSTR means you're investing in a legally established corporation. While highly concentrated in Bitcoin, the company has an existing software business, management team, and follows traditional financial reporting standards. For some, this corporate wrapper offers a perceived layer of familiarity or regulatory clarity compared to holding Bitcoin directly.
    * **Leveraged Bitcoin Exposure:** A significant portion of Strategy's corporate treasury is allocated to Bitcoin. Historically, Strategy has used debt financing to acquire large amounts of Bitcoin. This strategy can provide investors with a leveraged (though often more volatile) exposure to Bitcoin's price movements that might be difficult or costly to achieve through direct personal borrowing.
    * **Michael Saylor's Conviction:** Strategy's Executive Chairman, Michael Saylor, is a vocal and highly influential advocate for Bitcoin. His unwavering commitment to a Bitcoin-centric corporate treasury strategy provides a unique, ideologically-driven leadership that resonates with certain investors.

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
            total_cost = buy_amt + 15 + buy_amt * 0.002

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
            cash += proceeds - 15 - proceeds * 0.002
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
            # FIX: Corrected indentation for last_trade_date
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
    annualized_return = ((final_value / initial_capital) ** (1 / years) - 1) if years > 0 and initial_capital > 0 else 0

    result = {
        "Profit": profit,
        "Final Shares": shares,
        "Final Cash": cash,
        "Costs": costs,
        "Annualized Return": annualized_return * 100,  # Convert to percentage
        "Trades": len(trades_df),
        "Trades_df": trades_df,
        "Skipped_Trades": skipped_trades,
        "Portfolio_df": portfolio_df
    }

    return result

# Run backtest
# The backtest now uses @st.cache_data, so we don't need manual cache_key logic or run_backtest flag.
# Streamlit handles rerun detection based on function arguments.
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
            "Final Shares": final_shares,
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

    # --- MODIFIED PART FOR CONDITIONAL PROFIT COLOR IN TABLE (Pandas Styler) ---
    # Define a function to apply color based on profit value
    def highlight_profit_cell(val):
        color = "white" # Default for zero
        if val > 0:
            color = "green"
        elif val < 0:
            color = "red"
        return f"color: {color};"

    # Create a Styler object from the original yearly_metrics_df
    # Apply the color function to the 'Profit' column using applymap
    styled_yearly_metrics_df = yearly_metrics_df.style.applymap(
        highlight_profit_cell,
        subset=['Profit'] # Apply only to the 'Profit' column
    )

    # Apply number formatting to all relevant columns within the Styler
    # This formats the numbers *after* the color is applied.
    styled_yearly_metrics_df = styled_yearly_metrics_df.format({
        "Profit": "${:,.2f}",
        "Annual Return (%)": "{:.2f}%",
        "Final Cash": "${:,.2f}",
        "Total Costs": "${:,.2f}"
    })


    st.write("**Yearly Metrics**")
    # Pass the styled DataFrame to st.dataframe
    st.dataframe(styled_yearly_metrics_df, hide_index=True)
    st.write(f"**Total Profit Across All Years**: ${total_profit:,.2f}")
else:
    st.warning("No yearly metrics available. Check data span or trades.")
# --- END MODIFIED PART ---


# Display overall results
st.header("Overall Backtest Results")
col1, col2 = st.columns(2)

with col1:
    st.metric("Starting Capital", f"${initial_capital:,.2f}")

    # --- MODIFIED PART FOR CONDITIONAL PROFIT COLOR (Overall) ---
    profit_value = result['Profit']
    if profit_value >= 0:
        profit_color = "green"
        profit_sign = "+"
    else:
        profit_color = "red"
        profit_sign = "" # No sign for negative values
    st.markdown(
        f"""
        <div style="font-size: 14px; color: rgba(250, 250, 250, 0.6); margin-bottom: -15px;">Profit</div>
        <div style="font-size: 36px; font-weight: 600; color: {profit_color}; line-height: 1.2;">
            {profit_sign}${profit_value:,.2f}
        </div>
        """,
        unsafe_allow_html=True
    )
    # --- END MODIFIED PART ---

    st.metric("Annualized Return", f"{result['Annualized Return']:.2f}%")

with col2:
    st.metric("Final Shares", f"{result['Final Shares']:,.2f}")
    st.metric("Final Cash", f"${result['Final Cash']:,.2f}")
    st.metric("Total Costs", f"${result['Costs']:,.2f}")

st.metric("Number of Trades", result["Trades"])
if result["Skipped_Trades"] > 0:
    st.warning(f"**Skipped Trades**: {result['Skipped_Trades']} due to insufficient funds.")

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
def plot_price_path(price_col, title, color, trades_df, df):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df["Date"], df[price_col], label=title, color=color, alpha=0.7)

    if not trades_df.empty:
        buy_signals = trades_df[trades_df["Type"] == "Buy"]
        sell_signals = trades_df[trades_df["Type"] == "Sell"]

        if not buy_signals.empty:
            ax.scatter(buy_signals["Date"],
                       df.loc[df["Date"].isin(buy_signals["Date"]), price_col],
                       color="green", label="Buy", marker="^", s=100)

        if not sell_signals.empty:
            ax.scatter(sell_signals["Date"],
                       df.loc[df["Date"].isin(sell_signals["Date"]), price_col],
                       color="red", label="Sell", marker="v", s=100)

    # Add strategy info
    text_str = (
        f"Initial Capital: ${initial_capital:,.2f}\n"
        f"F&G Buy Threshold: {buy_threshold}\n"
        f"F&G Sell Threshold: {sell_threshold}\n"
        f"Buy Percentage: {buy_pct*100:.1f}%\n"
        f"Sell Percentage: {sell_pct*100:.1f}%\n"
        f"BTC Confirmation: {btc_confirm}\n"
        f"Cooldown Days: {cooldown_days}\n"
        f"Total Profit: ${result['Profit']:,.2f}"
    )

    ax.text(0.02, 0.98, text_str, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlabel("Date")
    ax.set_ylabel(f"{title.split()[0]} Price (USD)")
    ax.set_title(f"{title} with Trades")
    ax.legend()
    ax.grid(True)

    return fig

# Plot MSTR price path
st.subheader("MSTR Price Path")
if not df.empty:
    fig = plot_price_path("Adj Close", "MSTR Adj Close", "blue", result["Trades_df"], df)
    st.pyplot(fig)

    # Download button for the plot
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        label="Download MSTR Graph",
        data=buf,
        file_name="mstr_price_path.png",
        mime="image/png"
    )
    buf.close()
    plt.close(fig)
else:
    st.warning("No MSTR price data to plot.")

# Plot BTC price path
st.subheader("BTC Price Path")
if not df.empty:
    fig = plot_price_path("BTC_Close", "BTC Close", "orange", result["Trades_df"], df)
    st.pyplot(fig)

    # Download button for the plot
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        label="Download BTC Graph",
        data=buf,
        file_name="btc_price_path.png",
        mime="image/png"
    )
    buf.close()
    plt.close(fig)
else:
    st.warning("No BTC price data to plot.")
