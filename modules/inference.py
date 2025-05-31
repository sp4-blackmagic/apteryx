import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import time
import plotly.graph_objects as go

# Model leaderboard data
MODEL_DATA = [
    # task, model, r2, rmse, f1, mcc, time
    ["firmness", "CatBoost", 0.8321, 0.6898, 0.8208, 0.5951, 120.39],
    ["firmness", "XGBoost", 0.8175, 0.6887, 0.8194, 0.0394, 47.49],
    ["firmness", "LightGBM", 0.8029, 0.6987, 0.7987, 0.0579, 17.54],
    ["firmness", "Stacking", 0.7883, 0.5800, 0.7564, 0.0565, 34.03],
    ["firmness", "SVC_rbf", 0.7883, 0.6556, 0.7791, 0.2972, 0.60],
    ["firmness", "KNN", 0.7883, 0.6636, 0.7867, 0.0572, 0.00],
    ["firmness", "GradientBoosting", 0.7883, 0.6163, 0.7737, 0.0251, 67.94],
    ["firmness", "ExtraTrees", 0.7810, 0.6494, 0.7696, 0.3464, 0.49],
    ["firmness", "RandomForest", 0.7737, 0.6399, 0.7633, 0.3337, 2.68],
    ["firmness", "MLP", 0.7737, 0.6332, 0.7608, 0.0204, 2.15],
    ["firmness", "DecisionTree", 0.7226, 0.6246, 0.7296, 0.0057, 0.50],
    ["firmness", "AdaBoost", 0.7226, 0.5825, 0.7319, 0.3590, 8.78],
    ["firmness", "GaussianNB", 0.7153, 0.6025, 0.7417, 0.0220, 0.02],
    ["firmness", "SVC_linear", 0.6788, 0.5886, 0.7020, 0.0908, 0.73],
    ["firmness", "Ridge", 0.6569, 0.5544, 0.6703, 0.0083, 0.06],
    ["firmness", "LinearSVC", 0.6423, 0.5732, 0.6793, 0.0078, 4.17],
    ["firmness", "LogisticRegression", 0.6277, 0.5625, 0.6678, 0.0114, 1.15],
    ["firmness", "SGD", 0.6277, 0.5606, 0.6652, 0.0098, 0.97],
    ["firmness", "QDA", 0.6204, 0.4997, 0.6214, 0.0642, 0.12],
    ["firmness", "LDA", 0.3577, 0.3322, 0.3966, 0.0113, 0.27],
    ["ripeness", "LightGBM", 0.6131, 0.5943, 0.6033, 0.0453, 19.30],
    ["ripeness", "CatBoost", 0.5693, 0.5313, 0.5486, 0.5356, 119.41],
    ["ripeness", "XGBoost", 0.5401, 0.5091, 0.5208, 0.1165, 50.46],
    ["ripeness", "ExtraTrees", 0.5401, 0.5281, 0.5314, 0.4099, 0.63],
    ["ripeness", "GradientBoosting", 0.5255, 0.5164, 0.5191, 0.0806, 68.50],
    ["ripeness", "Ridge", 0.5182, 0.5176, 0.5182, 0.0078, 0.04],
    ["ripeness", "LinearSVC", 0.4964, 0.4946, 0.4940, 0.0066, 4.04],
    ["ripeness", "KNN", 0.4818, 0.4639, 0.4715, 0.6449, 0.00],
    ["ripeness", "RandomForest", 0.4672, 0.4615, 0.4536, 0.3385, 2.33],
    ["ripeness", "SVC_linear", 0.4672, 0.4173, 0.4300, 0.1010, 0.66],
    ["ripeness", "QDA", 0.4599, 0.3772, 0.4155, 0.0650, 0.11],
    ["ripeness", "LogisticRegression", 0.4526, 0.4621, 0.4472, 0.0102, 1.41],
    ["ripeness", "Stacking", 0.4453, 0.4225, 0.4266, 0.0883, 48.16],
    ["ripeness", "GaussianNB", 0.4453, 0.4344, 0.3961, 0.0326, 0.02],
    ["ripeness", "AdaBoost", 0.4307, 0.3829, 0.4042, 0.3079, 9.73],
    ["ripeness", "DecisionTree", 0.4234, 0.4318, 0.4262, 0.0075, 0.67],
    ["ripeness", "MLP", 0.4234, 0.3889, 0.3946, 0.0220, 0.60],
    ["ripeness", "SGD", 0.4234, 0.4278, 0.4238, 0.0096, 0.62],
    ["ripeness", "LDA", 0.3942, 0.3880, 0.3950, 0.0094, 0.29],
    ["ripeness", "SVC_rbf", 0.3723, 0.3480, 0.3503, 0.3394, 0.52],
]
MODEL_COLUMNS = ["Task", "Model", "R2", "RMSE", "F1", "MCC", "Time (s)"]
MODEL_DF = pd.DataFrame(MODEL_DATA, columns=MODEL_COLUMNS)

RIPENESS_LABELS = ["unripe", "perfect", "overripe"]
FIRMNESS_LABELS = ["too soft", "perfect", "too hard"]


def show_inference_page():
    st.header("🧠 Inference Engine")
    st.markdown("---")

    # 1. Preview from Camera Control
    st.subheader("1. Preview from Camera Control")
    camera_img = st.session_state.get("current_image")
    if camera_img is not None:
        st.image(camera_img, caption="Captured Kiwi Image", use_column_width=True)
        st.success("Image loaded from Camera Control.")
    else:
        st.info("No image captured yet. Please use the Camera Control page to take a picture.")

    st.markdown("---")
    # 2. Model Leaderboard & Ranking (collapsible)
    with st.expander("2. Model Leaderboard & Ranking", expanded=False):
        leaderboard_task = st.radio("Leaderboard Task", ["firmness", "ripeness"], horizontal=True, key="leaderboard_task")
        leaderboard_df = MODEL_DF[MODEL_DF["Task"]==leaderboard_task].sort_values("R2", ascending=False)
        st.dataframe(leaderboard_df, use_container_width=True, hide_index=True)
        st.plotly_chart(
            px.bar(leaderboard_df, x="Model", y="R2", color="Model", title=f"{leaderboard_task.capitalize()} Model R2 Scores", height=350),
            use_container_width=True
        )
        # Radar plot for top 5 models
        radar_metrics = ["R2", "RMSE", "F1", "MCC", "Time (s)"]
        top5 = leaderboard_df.head(5)
        fig_radar = go.Figure()
        for i, row in top5.iterrows():
            values = [row[m] if m != "Time (s)" else (1.0 / (row[m]+1e-6)) for m in radar_metrics]  # invert time for radar
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=radar_metrics,
                fill='toself',
                name=row["Model"]
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True,
            title=f"Radar Plot: Top 5 {leaderboard_task.capitalize()} Models (lower time is better)"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")
    st.subheader("3. Select Task and Model")
    # Task selection as radio
    task = st.radio("Prediction Task", ["firmness", "ripeness", "both"], horizontal=True)

    # Model selection as checkboxes
    st.markdown("<b>Select Model(s):</b>", unsafe_allow_html=True)
    model_options_firm = list(MODEL_DF[MODEL_DF["Task"]=="firmness"]["Model"].unique())
    model_options_ripe = list(MODEL_DF[MODEL_DF["Task"]=="ripeness"]["Model"].unique())
    selected_models_firm = []
    selected_models_ripe = []
    if task in ["firmness", "both"]:
        st.write("Firmness Models:")
        for m in model_options_firm:
            if st.checkbox(m, key=f"firm_{m}"):
                selected_models_firm.append(m)
    if task in ["ripeness", "both"]:
        st.write("Ripeness Models:")
        for m in model_options_ripe:
            if st.checkbox(m, key=f"ripe_{m}"):
                selected_models_ripe.append(m)

    st.markdown("---")
    st.subheader("4. Run Prediction")
    # Only allow prediction if image is available and at least one model is selected
    can_predict = camera_img is not None and ((selected_models_firm or selected_models_ripe))
    if not can_predict:
        st.warning("Please capture an image and select at least one model to run prediction.")
    else:
        if st.button("🥝 Run Prediction", type="primary", key="predict_btn"):
            with st.spinner("Running inference... (Simulated)"):
                time.sleep(2.0)
                results = {}
                confidences = {}
                if task in ["firmness", "both"]:
                    firm_idx = np.random.choice([0,1,2], p=[0.2,0.6,0.2])
                    firm_label = FIRMNESS_LABELS[firm_idx]
                    firm_conf = np.random.dirichlet(np.ones(3),size=1)[0]
                    results["firmness"] = firm_label
                    confidences["firmness"] = firm_conf
                if task in ["ripeness", "both"]:
                    ripe_idx = np.random.choice([0,1,2], p=[0.2,0.6,0.2])
                    ripe_label = RIPENESS_LABELS[ripe_idx]
                    ripe_conf = np.random.dirichlet(np.ones(3),size=1)[0]
                    results["ripeness"] = ripe_label
                    confidences["ripeness"] = ripe_conf
                st.session_state.infer_results = results
                st.session_state.infer_confidences = confidences
                st.session_state.pred_time = "2.00 seconds"
                st.session_state.prediction_done = True

    # Show prediction results
    if st.session_state.get("prediction_done", False):
        st.markdown("---")
        st.subheader("Prediction Results")
        # If multiple models, show confidence for each
        if (len(selected_models_firm) > 1 or len(selected_models_ripe) > 1):
            st.markdown("**Confidence for Each Selected Model**")
            cols = st.columns(max(len(selected_models_firm), len(selected_models_ripe), 2))
            idx = 0
            if len(selected_models_firm) > 1:
                for m in selected_models_firm:
                    conf = np.random.dirichlet(np.ones(3),size=1)[0]
                    cols[idx].plotly_chart(
                        px.bar(x=FIRMNESS_LABELS, y=conf, labels={"x":"Label","y":"Confidence"},
                               title=f"Firmness Confidence ({m})", color=FIRMNESS_LABELS, color_discrete_sequence=px.colors.qualitative.Pastel),
                        use_container_width=True
                    )
                    idx += 1
            if len(selected_models_ripe) > 1:
                for m in selected_models_ripe:
                    conf = np.random.dirichlet(np.ones(3),size=1)[0]
                    cols[idx].plotly_chart(
                        px.bar(x=RIPENESS_LABELS, y=conf, labels={"x":"Label","y":"Confidence"},
                               title=f"Ripeness Confidence ({m})", color=RIPENESS_LABELS, color_discrete_sequence=px.colors.qualitative.Pastel),
                        use_container_width=True
                    )
                    idx += 1
        # Original single-model logic
        cols = st.columns(2 if task!="both" else 3)
        idx = 0
        if "firmness" in st.session_state.infer_results:
            cols[idx].metric("Firmness", st.session_state.infer_results["firmness"])
            conf = st.session_state.infer_confidences["firmness"]
            cols[idx].plotly_chart(
                px.bar(x=FIRMNESS_LABELS, y=conf, labels={"x":"Label","y":"Confidence"},
                       title="Firmness Confidence", color=FIRMNESS_LABELS, color_discrete_sequence=px.colors.qualitative.Pastel),
                use_container_width=True
            )
            idx += 1
        if "ripeness" in st.session_state.infer_results:
            cols[idx].metric("Ripeness", st.session_state.infer_results["ripeness"])
            conf = st.session_state.infer_confidences["ripeness"]
            cols[idx].plotly_chart(
                px.bar(x=RIPENESS_LABELS, y=conf, labels={"x":"Label","y":"Confidence"},
                       title="Ripeness Confidence", color=RIPENESS_LABELS, color_discrete_sequence=px.colors.qualitative.Pastel),
                use_container_width=True
            )
            idx += 1
        cols[idx].info(f"Time to generate prediction: {st.session_state.pred_time}")

        st.markdown("---")
        st.subheader("5. Provide Real Label (Optional)")
        with st.form("real_label_form"):
            real_firm = None
            real_ripe = None
            if "firmness" in st.session_state.infer_results:
                real_firm = st.selectbox("True Firmness (if known)", FIRMNESS_LABELS, index=1, key="real_firm")
            if "ripeness" in st.session_state.infer_results:
                real_ripe = st.selectbox("True Ripeness (if known)", RIPENESS_LABELS, index=1, key="real_ripe")
            retrain = st.form_submit_button("Retrain Model with This Label (simulated)")
            if retrain:
                st.success("Thank you! The model will be retrained with your label (simulated).") 