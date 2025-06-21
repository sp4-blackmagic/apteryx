import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import time
import plotly.graph_objects as go
import httpx
import asyncio
import json

# TODO: update / make this more dynamic
INFERENCE_API_BASE_URL = "http://192.168.2.2:6969"  # Updated base API URL
STORAGE_API_URL = "http://192.168.2.11:8000/download_csv/" # Storage API URL



# Model leaderboard data (kept for leaderboard display)
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

async def get_model_types(api_base_url: str) -> list[str]:
    """
    Asynchronously fetch available model types from the API.
    """
    endpoint = f"{api_base_url}/model_types"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(endpoint)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        st.error(f"HTTP error occurred while fetching model types: {e}")
        return []
    except Exception as e:
        st.error(f"Error fetching model types: {e}")
        return []

async def call_inference(api_base_url: str, file_uid: str, models: list, storage_api_url: str) -> dict:
    """
    Asynchronously run inference via the API.
    """
    endpoint = f"{api_base_url}/evaluate/"
    
    try:
        payload = {
            "file_uid": file_uid,
            "models": models,
            "storage_api_url": storage_api_url
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()
            results = response.json()
            print(results) # Keep for debugging, can be removed later
            return results

    except httpx.HTTPStatusError as e:
        st.error(f"HTTP error occurred: {e.response.text if e.response else e}")
        # raise Exception(f"API request failed: {str(e)}") # Avoid raising to allow UI to handle
        return {"error": f"API request failed: {str(e)}", "details": e.response.text if e.response else "No response details"}
    except Exception as e:
        st.error(f"Error calling inference API: {e}")
        # raise Exception(f"Inference failed: {str(e)}") # Avoid raising
        return {"error": f"Inference failed: {str(e)}"}

def show_inference_page():
    st.header("🧠 Inference Engine")
    st.markdown("---")

    # Fetch model types
    if "available_models" not in st.session_state:
        st.session_state.available_models = []
    
    # Button to refresh model list
    if st.button("Refresh Model List"):
        api_url_for_models = st.session_state.get("inference_api_url", INFERENCE_API_BASE_URL)
        st.session_state.available_models = asyncio.run(get_model_types(api_url_for_models))
        if not st.session_state.available_models:
            st.warning("Could not fetch model list from the server. Please check the API URL or try again.")
        else:
            st.success(f"Fetched {len(st.session_state.available_models)} models.")
            st.rerun() # Rerun to update model selection UI

    # Initialize model list if empty and not yet fetched
    if not st.session_state.available_models:
        api_url_for_models = st.session_state.get("inference_api_url", INFERENCE_API_BASE_URL)
        st.session_state.available_models = asyncio.run(get_model_types(api_url_for_models))
        if not st.session_state.available_models:
            st.warning("Could not fetch model list initially. Try refreshing.")


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
    task = st.radio("Prediction Task", ["firmness", "ripeness", "both"], horizontal=True, key="task_selection")

    st.markdown("<b>Select Model(s):</b>", unsafe_allow_html=True)
    
    selected_models = []
    if not st.session_state.available_models:
        st.info("No models available for selection. Please refresh the model list or check API connectivity.")
    else:
        # Create columns for model selection for better layout if many models
        num_cols = 3 # Adjust as needed
        cols = st.columns(num_cols)
        for i, model_name in enumerate(st.session_state.available_models):
            if cols[i % num_cols].checkbox(model_name, key=f"model_{model_name}"):
                selected_models.append(model_name)
    
    # The old logic for selected_models_firm and selected_models_ripe is replaced
    # Now, selected_models contains all chosen models. The backend will handle task assignment.

    st.markdown("---")
    st.subheader("4. Run Prediction")

    # Only allow prediction if image is available and at least one model is selected
    can_predict = st.session_state.get("current_image") is not None and selected_models

    if not can_predict:
        st.warning("Please capture an image and select at least one model to run prediction.")
    else:
        if st.button("🥝 Run Prediction", type="primary", key="predict_btn"):
            start_time = time.time()
            
            #TODO: when storage API is ready, use the file_uid from session state
            # file_uid = st.session_state.get("current_file_uid", "sample_data") # Ensure this is set appropriately
            file_uid = "20250531171421960-e519a82232"

            api_base_url = st.session_state.get("inference_api_url", INFERENCE_API_BASE_URL)
            
            with st.spinner("Running inference..."):
                try:
                    api_result = asyncio.run(call_inference(api_base_url, file_uid, selected_models, STORAGE_API_URL))

                    # Raw API response will be stored in session state and displayed later
                    # st.subheader("Raw API Response:")
                    # st.json(api_result)
                    st.session_state.raw_api_result = api_result # Store for later display

                    print(f"API Result: {api_result}")  # Keep for debugging, can be removed later
                    
                    if "error" in api_result:
                        st.error(f"API Error: {api_result['error']}")
                        if "details" in api_result:
                            st.json(api_result['details'])
                        st.session_state.prediction_done = False # Ensure results are not shown
                        return # Stop further processing

                    # Initialize error collection
                    detailed_error_messages = []
                    
                    results = {}
                    confidences = {}
                    model_confidences = {"firmness": {}, "ripeness": {}}
                    
                    st.write("Selected prediction task:", task) # Display the selected task

                    if "results" in api_result and isinstance(api_result["results"], dict):
                        # Assuming the "0" key is consistent for the first (and likely only) item in batch
                        item_results = api_result["results"].get("0", {})
                        
                        for model_name_from_api, model_data in item_results.items():
                            if model_name_from_api not in selected_models: # Process only selected models' results
                                continue

                            # Process firmness predictions
                            if "firmness" in model_data and (task == "firmness" or task == "both"):
                                firm_data = model_data["firmness"]
                                if firm_data.get("status") == "success":
                                    if "firmness" not in results: # Take first successful model's prediction as primary
                                        results["firmness"] = firm_data["prediction_readable"]
                                        confidences["firmness"] = firm_data["prediction_proba"][0] if firm_data.get("prediction_proba") else [0,0,0]
                                    model_confidences["firmness"][model_name_from_api] = firm_data["prediction_proba"][0] if firm_data.get("prediction_proba") else [0,0,0]
                                elif firm_data.get("status") == "error":
                                    detailed_error_messages.append(f"Firmness ({model_name_from_api}): {firm_data.get('message', 'Unknown error')}")
                            
                            # Process ripeness predictions
                            if "ripeness_state" in model_data and (task == "ripeness" or task == "both"):
                                ripe_data = model_data["ripeness_state"]
                                if ripe_data.get("status") == "success":
                                    if "ripeness" not in results: # Take first successful model's prediction as primary
                                        results["ripeness"] = ripe_data["prediction_readable"]
                                        confidences["ripeness"] = ripe_data["prediction_proba"][0] if ripe_data.get("prediction_proba") else [0,0,0]
                                    model_confidences["ripeness"][model_name_from_api] = ripe_data["prediction_proba"][0] if ripe_data.get("prediction_proba") else [0,0,0]
                                elif ripe_data.get("status") == "error":
                                    detailed_error_messages.append(f"Ripeness ({model_name_from_api}): {ripe_data.get('message', 'Unknown error')}")
                        
                        st.subheader("Processed Data (before session state):")
                        
                        if results:
                            st.write("`results` (primary readable predictions):")
                            try:
                                results_df = pd.DataFrame(list(results.items()), columns=["Task", "Prediction"])
                                st.dataframe(results_df, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error displaying results table: {e}")
                                st.json(results) # Fallback to JSON
                        else:
                            st.write("`results`: No primary predictions.")

                        if confidences:
                            st.write("`confidences` (primary model probabilities):")
                            for task_name, probs in confidences.items():
                                st.markdown(f"**Task: {task_name}**")
                                labels = FIRMNESS_LABELS if task_name == "firmness" else RIPENESS_LABELS
                                if isinstance(probs, list) and len(probs) == len(labels):
                                    try:
                                        # Format probabilities
                                        formatted_probs = [f"{p*100:.2f}%" for p in probs]
                                        conf_df = pd.DataFrame({'Label': labels, 'Probability': formatted_probs})
                                        st.dataframe(conf_df, use_container_width=True)
                                    except Exception as e:
                                        st.error(f"Error displaying confidences table for {task_name}: {e}")
                                        st.json({task_name: probs}) # Fallback
                                else:
                                    st.write(f"Probabilities for {task_name}:")
                                    st.json(probs) # Fallback if not list or length mismatch
                        else:
                            st.write("`confidences`: No primary confidences.")

                        if model_confidences:
                            st.write("`model_confidences` (all model probabilities):")
                            for task_name, task_models in model_confidences.items():
                                if not task_models:
                                    st.markdown(f"**Task: {task_name}** - No model confidences.")
                                    continue
                                st.markdown(f"**Task: {task_name}**")
                                labels = FIRMNESS_LABELS if task_name == "firmness" else RIPENESS_LABELS
                                
                                table_data = []
                                for model_name, probs in task_models.items():
                                    if isinstance(probs, list) and len(probs) == len(labels):
                                        row = {'Model': model_name}
                                        for i, label in enumerate(labels):
                                            # Format probabilities
                                            row[label] = f"{probs[i]*100:.2f}%"
                                        table_data.append(row)
                                    else:
                                        st.write(f"Model: {model_name}, Probabilities for task {task_name}:")
                                        st.json(probs) # Fallback

                                if table_data:
                                    try:
                                        model_conf_df = pd.DataFrame(table_data)
                                        st.dataframe(model_conf_df, use_container_width=True)
                                    except Exception as e:
                                        st.error(f"Error displaying model_confidences table for {task_name}: {e}")
                                        st.json({task_name: task_models}) # Fallback
                                elif not task_models: # Already handled above, but as a safeguard
                                    pass # No data to show for this task if task_models was empty
                                elif task_models and not table_data: # task_models had items, but none were processable
                                    st.write(f"Could not format model confidences for task {task_name} into a table.")


                        else:
                            st.write("`model_confidences`: No model confidences.")
                        
                        st.session_state.infer_results = results
                        st.session_state.infer_confidences = confidences
                        st.session_state.model_confidences = model_confidences
                        st.session_state.pred_time = f"{time.time() - start_time:.2f} seconds"
                        st.session_state.prediction_done = True
                        
                        # Display errors or warnings based on prediction outcome
                        if not results: # No primary predictions were successful for any requested task
                            if detailed_error_messages:
                                st.error("No successful predictions were made. Specific errors encountered from models:")
                                for msg in detailed_error_messages:
                                    st.error(f"- {msg}")
                            else:
                                # This case means no models returned 'success' and no models returned 'error' with a message.
                                st.warning("No successful predictions were made for the selected task(s) and model(s). "
                                           "This could be due to model incompatibility with the task, issues processing the input data, "
                                           "or server-side configuration problems. Check API logs for more details.")
                        elif detailed_error_messages: # Some primary predictions were successful, but some models/tasks had errors
                            st.warning("Partial success: Primary predictions are available for some tasks, but some models/tasks encountered errors.")
                            st.warning("Details of errors reported by models:")
                            for msg in detailed_error_messages:
                                st.warning(f"- {msg}")
                        # If results is not empty and detailed_error_messages is empty, it's a full success, no extra message needed here.
                        
                        # st.rerun() # This was likely causing the results to blink and disappear
                    else:
                        st.error("Invalid response format from API or no results found.")
                        st.json(api_result) 
                
                except Exception as e:
                    st.error(f"Error during inference: {str(e)}")
                    st.session_state.prediction_done = False


    if st.session_state.get("prediction_done", False):
        st.markdown("---")
        st.subheader("Prediction Results")
        
        model_confidences = st.session_state.get("model_confidences", {"firmness": {}, "ripeness": {}})
        infer_results = st.session_state.get("infer_results", {})

        # Display individual model confidences if multiple models were involved for a task
        # This logic needs to know which models were selected for which task, or show all selected models' confidences
        
        # Simplified: Show confidence for all selected models that returned results
        # This part might need refinement based on how `selected_models` (which is now a flat list)
        # maps to tasks if a user selects "both" tasks.
        # For now, we iterate through `model_confidences` which is already populated correctly.

        num_selected_displayable_firmness = len(model_confidences.get("firmness", {}))
        num_selected_displayable_ripeness = len(model_confidences.get("ripeness", {}))

        if num_selected_displayable_firmness > 1 or num_selected_displayable_ripeness > 1 :
            st.markdown("**Confidence for Each Successful Model**")
            
            # Determine max columns needed
            max_cols_needed = 0
            if task in ["firmness", "both"] and num_selected_displayable_firmness > 0:
                max_cols_needed = max(max_cols_needed, num_selected_displayable_firmness)
            if task in ["ripeness", "both"] and num_selected_displayable_ripeness > 0:
                 max_cols_needed = max(max_cols_needed, num_selected_displayable_ripeness)
            
            if max_cols_needed == 0 and (num_selected_displayable_firmness > 0 or num_selected_displayable_ripeness > 0) : # if only one model gave result for one task
                 pass # will be handled by single model logic
            elif max_cols_needed > 0:
                cols = st.columns(max_cols_needed if max_cols_needed > 1 else 2) # Ensure at least 2 cols if section is shown
                col_idx = 0

                if task in ["firmness", "both"] and model_confidences.get("firmness"):
                    for model_name, conf_values in model_confidences["firmness"].items():
                        if col_idx < len(cols): # Check to prevent IndexError if too many models for columns
                            cols[col_idx].plotly_chart(
                                px.bar(x=FIRMNESS_LABELS, y=conf_values, labels={"x":"Label","y":"Confidence"},
                                    title=f"Firmness ({model_name})", color=FIRMNESS_LABELS, 
                                    color_discrete_sequence=px.colors.qualitative.Pastel),
                                use_container_width=True
                            )
                            col_idx +=1
                        else: # Fallback if more models than columns
                            st.plotly_chart(px.bar(x=FIRMNESS_LABELS, y=conf_values, labels={"x":"Label","y":"Confidence"}, title=f"Firmness ({model_name})", color=FIRMNESS_LABELS, color_discrete_sequence=px.colors.qualitative.Pastel), use_container_width=True)


                col_idx = 0 # Reset for ripeness if firmness was also shown in same row of columns
                if task in ["ripeness", "both"] and model_confidences.get("ripeness"):
                    for model_name, conf_values in model_confidences["ripeness"].items():
                        if col_idx < len(cols):
                            cols[col_idx].plotly_chart(
                                px.bar(x=RIPENESS_LABELS, y=conf_values, labels={"x":"Label","y":"Confidence"},
                                    title=f"Ripeness ({model_name})", color=RIPENESS_LABELS,
                                    color_discrete_sequence=px.colors.qualitative.Pastel),
                                use_container_width=True
                            )
                            col_idx +=1
                        else:
                             st.plotly_chart(px.bar(x=RIPENESS_LABELS, y=conf_values, labels={"x":"Label","y":"Confidence"}, title=f"Ripeness ({model_name})", color=RIPENESS_LABELS, color_discrete_sequence=px.colors.qualitative.Pastel), use_container_width=True)


        # Display primary prediction (from the first successful model for each task)
        display_cols = st.columns(3) # For Firmness, Ripeness, Time
        current_col = 0
        
        if "firmness" in infer_results:
            display_cols[current_col].metric("Firmness Prediction", infer_results["firmness"])
            if "firmness" in st.session_state.infer_confidences:
                conf = st.session_state.infer_confidences["firmness"]
                display_cols[current_col].plotly_chart(
                    px.bar(x=FIRMNESS_LABELS, y=conf, labels={"x":"Label","y":"Confidence"},
                           title="Overall Firmness Confidence", color=FIRMNESS_LABELS, color_discrete_sequence=px.colors.qualitative.Pastel),
                    use_container_width=True
                )
            current_col += 1
        
        if "ripeness" in infer_results:
            display_cols[current_col].metric("Ripeness Prediction", infer_results["ripeness"])
            if "ripeness" in st.session_state.infer_confidences:
                conf = st.session_state.infer_confidences["ripeness"]
                display_cols[current_col].plotly_chart(
                    px.bar(x=RIPENESS_LABELS, y=conf, labels={"x":"Label","y":"Confidence"},
                           title="Overall Ripeness Confidence", color=RIPENESS_LABELS, color_discrete_sequence=px.colors.qualitative.Pastel),
                    use_container_width=True
                )
            current_col += 1

        if current_col < len(display_cols): # Check if space for time
             display_cols[current_col].info(f"Time to generate prediction: {st.session_state.pred_time}")
        else: # if no space, print it below
             st.info(f"Time to generate prediction: {st.session_state.pred_time}")


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
        
        # Display Raw API Response at the end
        if "raw_api_result" in st.session_state:
            with st.expander("Raw API Response", expanded=False):
                st.json(st.session_state.raw_api_result)