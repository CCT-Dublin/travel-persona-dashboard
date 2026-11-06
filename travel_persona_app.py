import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, norm

sns.set(style="whitegrid")

# -----------------------------
# Run history storage
# -----------------------------
if "run_history" not in st.session_state:
    st.session_state.run_history = []

# -----------------------------
# Simulation function
# -----------------------------
def simulate_travel_data(
    n=200,
    p_cancel=0.25,
    price_mu=1200,
    price_sigma=250,
    sat_mu=8,
    sat_sigma=1.2
):
    personas = ["Budget Traveller", "Luxury Traveller", "Family Traveller", "Solo Explorer", "Retired Traveller"]
    destinations = ["Paris", "Rome", "Barcelona", "Berlin", "Lisbon"]
    packages = ["City Break", "Beach", "Cultural Tour"]
    seasons = ["High", "Shoulder", "Low"]
    age_groups = ["18-30", "31-50", "51+"]

    persona_choice = np.random.choice(personas, n)
    destination_choice = np.random.choice(destinations, n)
    package_choice = np.random.choice(packages, n)
    season_choice = np.random.choice(seasons, n)
    age_choice = np.random.choice(age_groups, n)

    price = []
    for p in persona_choice:
        if p == "Budget Traveller":
            price.append(np.random.normal(price_mu * 0.6, price_sigma * 0.6))
        elif p == "Luxury Traveller":
            price.append(np.random.normal(price_mu * 1.5, price_sigma * 1.0))
        elif p == "Family Traveller":
            price.append(np.random.normal(price_mu * 1.1, price_sigma * 0.8))
        elif p == "Solo Explorer":
            price.append(np.random.normal(price_mu * 0.9, price_sigma * 0.7))
        else:
            price.append(np.random.normal(price_mu * 1.0, price_sigma * 0.7))

    price = np.round(price, 2)

    booking_status = np.random.choice(["Booked", "Cancelled"], n, p=[1 - p_cancel, p_cancel])

    satisfaction = []
    for status in booking_status:
        if status == "Booked":
            satisfaction.append(np.random.normal(sat_mu, sat_sigma))
        else:
            satisfaction.append(np.nan)

    satisfaction = np.round(satisfaction, 2)

    df = pd.DataFrame({
        "Persona": persona_choice,
        "Destination": destination_choice,
        "Package": package_choice,
        "Season": season_choice,
        "Age Group": age_choice,
        "Price": price,
        "Booking Status": booking_status,
        "Satisfaction": satisfaction
    })

    return df

# -----------------------------
# Streamlit app
# -----------------------------
def main():
    st.set_page_config(page_title="Travel Agency Persona Simulation App", layout="wide")

    # -----------------------------
    # Custom CSS (burgundy sidebar + slider colour)
    # -----------------------------
    st.markdown("""
<style>

    /* Slider track (burgundy) */
    input[type=range]::-webkit-slider-runnable-track {
        background: #5A0F2C !important;   /* burgundy */
        height: 6px;
    }

    /* Slider thumb (white) */
    input[type=range]::-webkit-slider-thumb {
        background: #ffffff !important;
        border: 2px solid #5A0F2C !important;
        height: 16px;
        width: 16px;
        border-radius: 50%;
        margin-top: -5px;
    }

    /* Firefox slider track */
    input[type=range]::-moz-range-track {
        background: #5A0F2C !important;
        height: 6px;
    }

    /* Firefox slider thumb */
    input[type=range]::-moz-range-thumb {
        background: #ffffff !important;
        border: 2px solid #5A0F2C !important;
        height: 16px;
        width: 16px;
        border-radius: 50%;
    }

</style>
""", unsafe_allow_html=True)

    st.title("✈️ Travel Agency Persona Simulation App")

    st.sidebar.header("Simulation Parameters")
    sample_size = st.sidebar.slider("Sample size", 50, 1000, 200, 50)
    cancel_tolerance = st.sidebar.slider("Expected cancellation rate", 0.05, 0.5, 0.25, 0.05)
    price_mu = st.sidebar.slider("Average base price (€)", 500, 2000, 1200, 100)
    price_sigma = st.sidebar.slider("Price standard deviation", 50, 400, 250, 25)
    sat_mu = st.sidebar.slider("Average satisfaction (1–10)", 5.0, 9.5, 8.0, 0.1)
    sat_sigma = st.sidebar.slider("Satisfaction standard deviation", 0.5, 2.5, 1.2, 0.1)

    if st.sidebar.button("Run simulation"):
        df = simulate_travel_data(
            n=sample_size,
            p_cancel=cancel_tolerance,
            price_mu=price_mu,
            price_sigma=price_sigma,
            sat_mu=sat_mu,
            sat_sigma=sat_sigma
        )

        # -----------------------------
        # Add to run history
        # -----------------------------
        st.session_state.run_history.append({
            "Sample size": sample_size,
            "Cancel tolerance": cancel_tolerance,
            "Price μ": price_mu,
            "Price σ": price_sigma,
            "Sat μ": sat_mu,
            "Sat σ": sat_sigma
        })

        st.subheader("🧳 Passenger Manifest — Sample of Simulated Data")
        st.dataframe(df.head())

        # -----------------------------
        # Descriptive statistics
        # -----------------------------
        st.subheader("🌍 Destination Insights — Descriptive Statistics")

        mean_price = df["Price"].mean()
        median_price = df["Price"].median()
        std_price = df["Price"].std()
        var_price = df["Price"].var()

        sat_series = df["Satisfaction"].dropna()
        mean_sat = sat_series.mean()
        median_sat = sat_series.median()
        std_sat = sat_series.std()
        var_sat = sat_series.var()

        p_cancelled_emp = (df["Booking Status"] == "Cancelled").mean()
        p_luxury = (df["Persona"] == "Luxury Traveller").mean()

        stat_price, p_price = shapiro(df["Price"])
        stat_sat, p_sat = shapiro(sat_series) if len(sat_series) > 3 else (np.nan, np.nan)

        desc_df = pd.DataFrame({
            "Metric": [
                "Mean price", "Median price", "Std price", "Var price",
                "Mean satisfaction", "Median satisfaction", "Std satisfaction", "Var satisfaction",
                "Empirical cancellation rate", "Proportion Luxury Traveller",
                "Shapiro p-value (Price)", "Shapiro p-value (Satisfaction)"
            ],
            "Value": [
                mean_price, median_price, std_price, var_price,
                mean_sat, median_sat, std_sat, var_sat,
                p_cancelled_emp, p_luxury,
                p_price, p_sat
            ]
        })

        st.table(desc_df)

        # -----------------------------
        # Visualisations
        # -----------------------------
        st.subheader("📸 Window View — Visualisations")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Price distribution (histogram + KDE)**")
            fig1, ax1 = plt.subplots()
            sns.histplot(df["Price"], kde=True, ax=ax1, color="#8A1F47")
            st.pyplot(fig1)

        with col2:
            st.markdown("**Boxplot of price by persona (custom colours)**")
            fig2, ax2 = plt.subplots()
            palette = ["#8A1F47", "#E07A5F", "#3D405B", "#81B29A", "#F2CC8F"]
            sns.boxplot(data=df, x="Persona", y="Price", ax=ax2, palette=palette)
            ax2.tick_params(axis='x', rotation=45)
            st.pyplot(fig2)

        st.markdown("**Cancellation rate by season (custom colours)**")
        season_cancel = df.groupby("Season")["Booking Status"].apply(lambda x: (x == "Cancelled").mean()).reset_index()
        fig3, ax3 = plt.subplots()
        sns.barplot(data=season_cancel, x="Season", y="Booking Status", ax=ax3,
                    palette=["#8A1F47", "#E07A5F", "#81B29A"])
        st.pyplot(fig3)

        # -----------------------------
        # Outlier removal
        # -----------------------------
        st.subheader("⚠️ Turbulence Check — Outlier Removal (IQR)")

        Q1 = df["Price"].quantile(0.25)
        Q3 = df["Price"].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df_no_outliers = df[(df["Price"] >= lower) & (df["Price"] <= upper)]

        outlier_df = pd.DataFrame({
            "Metric": ["Rows before", "Rows after"],
            "Value": [df.shape[0], df_no_outliers.shape[0]]
        })

        st.table(outlier_df)

        # -----------------------------
        # Bayes theorem
        # -----------------------------
        st.subheader("🔐 Security Gate — Bayes Theorem")

        p_lux = (df["Persona"] == "Luxury Traveller").mean()
        p_booked = (df["Booking Status"] == "Booked").mean()
        lux_df = df[df["Persona"] == "Luxury Traveller"]
        p_booked_given_lux = (lux_df["Booking Status"] == "Booked").mean() if len(lux_df) > 0 else np.nan

        if p_booked > 0 and not np.isnan(p_booked_given_lux):
            p_lux_given_booked = (p_booked_given_lux * p_lux) / p_booked
        else:
            p_lux_given_booked = np.nan

        bayes_df = pd.DataFrame({
            "Quantity": ["P(Luxury)", "P(Booked)", "P(Booked | Luxury)", "P(Luxury | Booked)"],
            "Value": [p_lux, p_booked, p_booked_given_lux, p_lux_given_booked]
        })

        st.table(bayes_df)

        # -----------------------------
        # p-norm probability tests
        # -----------------------------
        st.subheader("📈 Flight Probability — p-norm Tests")

        mu_price = df["Price"].mean()
        sigma_price = df["Price"].std()

        prob_price_high = 1 - norm.cdf(1500, mu_price, sigma_price)

        mu_sat = sat_series.mean()
        sigma_sat = sat_series.std()
        prob_sat_low = norm.cdf(7, mu_sat, sigma_sat)

        prob_df = pd.DataFrame({
            "Question": ["P(Price > 1500)", "P(Satisfaction < 7)"],
            "Probability": [prob_price_high, prob_sat_low]
        })

        st.table(prob_df)

        # -----------------------------
        # Tolerance checks
        # -----------------------------
        st.subheader("🧭 Tolerance Checks — Price, Cancellation, Satisfaction")

        price_lower_tol = mu_price - 2 * sigma_price
        price_upper_tol = mu_price + 2 * sigma_price
        in_price_tol = ((df["Price"] >= price_lower_tol) & (df["Price"] <= price_upper_tol)).mean()
        out_price_tol = 1 - in_price_tol

        emp_cancel = p_cancelled_emp
        cancel_diff = emp_cancel - cancel_tolerance

        sat_out_tol = (sat_series < 7).mean()
        sat_in_tol = 1 - sat_out_tol

        tol_df = pd.DataFrame({
            "Metric": [
                "Price in ±2σ tolerance",
                "Price out of ±2σ tolerance",
                "Empirical cancellation rate",
                "Expected cancellation tolerance",
                "Cancellation difference (emp - tol)",
                "Satisfaction in tolerance (>=7)",
                "Satisfaction out of tolerance (<7)"
            ],
            "Value": [
                in_price_tol,
                out_price_tol,
                emp_cancel,
                cancel_tolerance,
                cancel_diff,
                sat_in_tol,
                sat_out_tol
            ]
        })

        st.table(tol_df)

        # -----------------------------
        # Run history
        # -----------------------------
        st.subheader("🛩️ Flight Log — Run History")

        st.dataframe(pd.DataFrame(st.session_state.run_history))

    else:
        st.info("Use the controls in the sidebar and click **Run simulation** to generate the app output.")

if __name__ == "__main__":
    main()