import streamlit as st
import pandas as pd
import numpy as np
import os
import altair as alt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="PSO Multi-Objective Optimization", layout="wide")

st.title("🚍 Multi-Objective Optimization using PSO")
st.write("Objectives: **Minimize Distance and Fare**")

# =========================
# AUTO LOAD DATASET
# =========================
DATA_PATH = "delhi_metro_updated.csv"

if not os.path.exists(DATA_PATH):
    st.error("dataset.csv not found in project folder.")
    st.stop()

data = pd.read_csv(DATA_PATH)
data.columns = data.columns.str.lower().str.replace(" ", "_")

required_cols = ["distance_km", "fare", "cost_per_passenger", "passengers"]
if not all(col in data.columns for col in required_cols):
    st.error("Dataset must contain: Distance_km, Fare, Cost_per_passenger, Passengers")
    st.stop()

distance = data["distance_km"].values
fare = data["fare"].values
n = len(distance)

# =========================
# SIDEBAR PARAMETERS
# =========================
st.sidebar.header("⚙ PSO Parameters")

particles = st.sidebar.slider("Number of Particles", 10, 100, 30)
iterations = st.sidebar.slider("Iterations", 10, 300, 100)

w = st.sidebar.slider("Inertia Weight (w)", 0.1, 0.9, 0.5)
c1 = st.sidebar.slider("Cognitive Coefficient (c1)", 0.5, 3.0, 1.5)
c2 = st.sidebar.slider("Social Coefficient (c2)", 0.5, 3.0, 1.5)

st.sidebar.subheader("Objective Weights")
w_distance = st.sidebar.slider("Distance Weight", 0.0, 1.0, 0.6)
w_fare = st.sidebar.slider("Fare Weight", 0.0, 1.0, 0.4)

if w_distance + w_fare == 0:
    st.sidebar.error("At least one weight must be > 0")
    st.stop()

# =========================
# FITNESS FUNCTION
# =========================
def fitness(i):
    return w_distance * distance[i] + w_fare * fare[i]

# =========================
# PSO TRAINING FUNCTION
# =========================
def pso_train():
    # Initialize particles (positions are indices)
    positions = np.random.randint(0, n, particles)
    velocities = np.zeros(particles)

    pbest_positions = positions.copy()
    pbest_scores = np.array([fitness(i) for i in positions])

    gbest_index = pbest_positions[np.argmin(pbest_scores)]
    gbest_score = min(pbest_scores)

    convergence = []

    for _ in range(iterations):
        for i in range(particles):
            r1, r2 = np.random.rand(), np.random.rand()

            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest_positions[i] - positions[i])
                + c2 * r2 * (gbest_index - positions[i])
            )

            positions[i] = int(np.clip(round(positions[i] + velocities[i]), 0, n - 1))

            score = fitness(positions[i])

            if score < pbest_scores[i]:
                pbest_scores[i] = score
                pbest_positions[i] = positions[i]

        gbest_index = pbest_positions[np.argmin(pbest_scores)]
        gbest_score = min(pbest_scores)
        convergence.append(gbest_score)

    return gbest_index, gbest_score, convergence

# =========================
# PARETO FRONT
# =========================
def pareto_front(dist, cost):
    pareto = []
    for i in range(len(dist)):
        dominated = False
        for j in range(len(dist)):
            if (
                dist[j] <= dist[i]
                and cost[j] <= cost[i]
                and (dist[j] < dist[i] or cost[j] < cost[i])
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(i)
    return pareto

pareto_idx = pareto_front(distance, fare)

# =========================
# RUN PSO
# =========================
if st.button("▶ Run PSO Optimization"):
    with st.spinner("Training PSO..."):
        best_idx, best_score, convergence = pso_train()

    st.success("Optimization Completed ✅")

    # =========================
    # RESULTS
    # =========================
    col1, col2, col3 = st.columns(3)
    col1.metric("Distance (km)", distance[best_idx])
    col2.metric("Fare", fare[best_idx])
    col3.metric("Fitness Score", round(best_score, 3))

    st.subheader("📊 Additional Information")
    st.write("Cost per Passenger:", data.loc[best_idx, "cost_per_passenger"])
    st.write("Passengers:", data.loc[best_idx, "passengers"])

    # =========================
    # CONVERGENCE
    # =========================
    st.subheader("📉 Convergence Curve")
    st.line_chart(convergence)

    # =========================
    # PARETO FRONT
    # =========================
    st.subheader("📈 Pareto Front (Distance vs Fare)")

    plot_df = pd.DataFrame({
        "Distance": distance,
        "Fare": fare,
        "Type": [
            "Pareto-optimal" if i in pareto_idx else "Dominated"
            for i in range(len(distance))
        ]
    })

    chart = alt.Chart(plot_df).mark_circle(size=80).encode(
        x="Distance",
        y="Fare",
        color=alt.Color(
            "Type",
            scale=alt.Scale(domain=["Pareto-optimal", "Dominated"],
                            range=["red", "lightgray"]),
            legend=alt.Legend(title="Solution Type")
        ),
        tooltip=["Distance", "Fare", "Type"]
    )

    st.altair_chart(chart, use_container_width=True)
    st.info("🔴 Red points represent Pareto-optimal solutions.")

    st.subheader("📄 Dataset Preview")
    st.dataframe(data.head())
