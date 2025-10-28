import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- Core functions --------------------------------------------------


def calc_stamp_duty(price: float) -> float:
    """Calculate UK residential stamp duty (England & NI bands, 2025)."""
    brackets = [
        (250000, 0.00),
        (925000, 0.05),
        (1500000, 0.10),
        (float("inf"), 0.12),
    ]
    duty = 0.0
    prev = 0.0
    for limit, rate in brackets:
        taxable = min(price, limit) - prev
        if taxable > 0:
            duty += taxable * rate
        prev = limit
        if price <= limit:
            break
    return duty


def mortgage_monthly_payment(principal, annual_rate, years):
    r = annual_rate / 12.0
    n = years * 12
    if r == 0:
        return principal / n
    return principal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)


def simulate(
    purchase_price,
    house_price_growth_annual,
    mortgage_rate_annual,
    mortgage_years,
    deposit,
    rent_pcm,
    invest_return_annual,
    include_stamp_duty=True,
    include_cgt=True,
):
    months = mortgage_years * 12
    stamp_duty = calc_stamp_duty(purchase_price) if include_stamp_duty else 0.0

    borrowed = purchase_price - deposit
    m_payment = mortgage_monthly_payment(
        borrowed, mortgage_rate_annual, mortgage_years
    )

    monthly_mortgage_rate = mortgage_rate_annual / 12.0
    monthly_house_growth = house_price_growth_annual / 12.0
    monthly_invest_return = (
        invest_return_annual / 12.0 * (1 - 0.20 if include_cgt else 1.0)
    )

    month_idx = np.arange(1, months + 1, dtype=int)
    remaining = np.zeros(months + 1)
    remaining[0] = borrowed
    savings = np.zeros(months + 1)
    house_val = np.zeros(months + 1)
    equity_buy = np.zeros(months + 1)

    house_val[0] = purchase_price

    # Adjust initial capital if stamp duty is included
    initial_investment = (
        deposit + stamp_duty if include_stamp_duty else deposit
    )
    savings[1] = (
        m_payment - rent_pcm * (1 + rent_growth / 12)
    ) + initial_investment

    # First month
    interest_first = borrowed * monthly_mortgage_rate
    principal_first = m_payment - interest_first
    remaining[1] = borrowed - principal_first
    house_val[1] = purchase_price * (1 + monthly_house_growth) ** 1
    equity_buy[1] = house_val[1] - remaining[1]

    for m in range(2, months + 1):
        savings[m] = savings[m - 1] * (1 + monthly_invest_return) + (
            m_payment - rent_pcm * (1 + rent_growth / 12) ** m
        )
        interest = remaining[m - 1] * monthly_mortgage_rate
        principal_paid = m_payment - interest
        remaining[m] = max(remaining[m - 1] - principal_paid, 0.0)
        house_val[m] = purchase_price * (1 + monthly_house_growth) ** m
        equity_buy[m] = house_val[m] - remaining[m]

    df = pd.DataFrame(
        {
            "month": month_idx,
            "remaining_mortgage": remaining[1:],
            "house_value": house_val[1:],
            "equity_buy": equity_buy[1:],
            "savings_rent": savings[1:],
        }
    )
    df["diff_buy_minus_rent"] = df["equity_buy"] - df["savings_rent"]
    df["breakeven_reached"] = df["diff_buy_minus_rent"] >= 0

    return df, m_payment, borrowed, stamp_duty


def first_breakeven_month(df):
    hits = df[df["breakeven_reached"]]
    if len(hits) == 0:
        return None
    return int(hits.iloc[0]["month"])


# --- Streamlit UI ------------------------------------------------------

st.set_page_config(page_title="Buy vs Rent Simulator", layout="centered")
st.title("Buy vs Rent Simulator (with Stamp Duty)")

st.write(
    "Compare the long-term financial outcomes of buying versus renting, "
    "now including UK stamp duty as an optional up-front cost."
)

st.header("Assumptions")
col1, col2 = st.columns(2)

with col1:
    purchase_price = st.slider(
        "Purchase price (£)", 200_000, 2_000_000, 1_000_000, 10_000
    )
    deposit = st.slider("Deposit (£)", 0, int(purchase_price), 200_000, 10_000)
    mortgage_rate = (
        st.slider("Mortgage rate (annual %)", 0.0, 10.0, 4.5, 0.1) / 100
    )
    mortgage_years = st.slider("Mortgage length (years)", 5, 40, 20, 1)
    include_stamp_duty = st.checkbox("Include stamp duty (UK rules)", True)

with col2:
    house_growth = (
        st.slider("House price growth (annual %)", -5.0, 10.0, 3.0, 0.1) / 100
    )
    rent_pcm = st.slider(
        "Rent (per calendar month, £)", 500, 10_000, 2_500, 100
    )
    rent_growth = (
        st.slider("Rent increase (annual %)", 0.0, 10.0, 4.5, 0.1) / 100
    )
    invest_return = (
        st.slider("Investment return (annual %)", 0.0, 15.0, 8.0, 0.5) / 100
    )
    include_cgt = st.checkbox("Include capital gains (20%)", True)


df, monthly_payment, borrowed, stamp_duty = simulate(
    purchase_price,
    house_growth,
    mortgage_rate,
    mortgage_years,
    deposit,
    rent_pcm,
    invest_return,
    include_stamp_duty,
    include_cgt,
)

breakeven_month = first_breakeven_month(df)

# --- Metrics summary ---------------------------------------------------

st.subheader("Key outputs")

colA, colB, colC, colD = st.columns(4)
colA.metric("Borrowed (£)", f"{borrowed:,.0f}")
colB.metric("Monthly mortgage (£)", f"{monthly_payment:,.0f}")
colC.metric(
    "Stamp duty (£)",
    f"{stamp_duty:,.0f}" if include_stamp_duty else "Excluded",
)
if breakeven_month:
    colD.metric(
        "Breakeven", f"Month {breakeven_month} (~{breakeven_month/12:.1f} yrs)"
    )
else:
    colD.metric("Breakeven", "Never")

# --- Plot --------------------------------------------------------------

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=df["month"],
        y=df["equity_buy"],
        mode="lines",
        name="Equity if buying (£)",
        hovertemplate="Month %{x}<br>Net worth £%{y:,.0f}<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=df["month"],
        y=df["savings_rent"],
        mode="lines",
        name="Savings if renting (£)",
        hovertemplate="Month %{x}<br>Net worth £%{y:,.0f}<extra></extra>",
    )
)
if breakeven_month:
    fig.add_vline(
        x=breakeven_month,
        line_width=1,
        line_dash="dash",
        line_color="grey",
    )
fig.update_layout(
    title="Buy vs Rent Over Time",
    xaxis_title="Month",
    yaxis_title="Net worth (£)",
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Month-by-month breakdown")
st.dataframe(df)
