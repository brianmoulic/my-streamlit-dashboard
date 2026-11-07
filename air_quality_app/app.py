import json

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Air Quality Advisory Support", layout="wide")

ASSETS = "final_assets"
MODEL_FILES = {
    "Decision Tree": "decision_tree.joblib",
    "Bagging": "bagging.joblib",
    "Random Forest": "random_forest.joblib",
    "Gradient Boosting": "gradient_boosting.joblib",
    "AdaBoost": "adaboost.joblib",
}
RECOMMENDED = "Bagging"
CLASS_ORDER = ["Good", "Moderate", "Unhealthy"]

FEATURES = [
    "city_name", "coord.lat", "coord.lon", "month", "day_of_week",
    "season", "prev_day_aqi", "prev_day_category", "aqi_7day_avg",
]

POLLUTANTS = {
    "components.pm2_5": "PM2.5",
    "components.pm10": "PM10",
    "components.no2": "NO2",
    "components.o3": "O3",
    "components.so2": "SO2",
    "components.co": "CO",
    "components.no": "NO",
    "components.nh3": "NH3",
}

CATEGORY_MEANING = {
    "Good": "Good (AQI 1–2): Air quality is generally satisfactory, and pollution poses little risk to most people.",
    "Moderate": "Moderate (AQI 3): Air quality is acceptable for most people, but sensitive groups such as children, the elderly, and people with respiratory conditions may be affected.",
    "Unhealthy": "Unhealthy (AQI 4–5): Pollution levels may affect the health of the general public, and sensitive groups may be affected more seriously.",
}

ACTIONS = {
    "Good": {
        "headline": "Decision: No advisory needed today. Continue routine monitoring and stay ready.",
        "meaning": "Air quality is expected to be within safe levels, so there is no need to alert the public. The main job today is making sure the LGU can react quickly if the forecast changes.",
        "now": [
            "Do not issue an advisory. Keep public messages normal to avoid unnecessary alarm.",
            "Check that the advisory templates and the contact list (city health office, barangay officials, schools) are up to date, so an advisory can go out the same day if needed.",
            "Write today's predicted category in the monitoring log.",
        ],
        "monitoring_title": "How to do routine monitoring",
        "monitoring": [
            "Daily check: open this app once a day and note the predicted category and its probabilities.",
            "Trend check: look at the 7-day average AQI on the Data Insights tab. A rise over several days is an early sign of trouble even while the category is still Good.",
            "Accuracy check: when the day's readings come in, compare the recorded category with the prediction and log any mismatch.",
            "Weekly review: summarize the week's categories and share them with the head of the environmental office.",
        ],
        "escalate": "Move to the Moderate steps if the next prediction is Moderate, or if the 7-day average keeps rising for several days.",
    },
    "Moderate": {
        "headline": "Decision: Prepare an early-warning advisory for sensitive groups.",
        "meaning": "Air quality may worsen. Children, the elderly, and people with respiratory conditions are the most likely to be affected first.",
        "now": [
            "Send a precautionary notice to the city health office and barangay health workers so they are ready if conditions get worse.",
            "Prepare the advisory for sensitive groups (what to avoid, when to stay indoors, who to contact) and agree on who approves it, so it can be released quickly.",
            "Inform schools and day-care centers so they can plan outdoor activities.",
            "Review possible preventive measures, such as traffic measures in busy areas, with the traffic office, but do not apply restrictions yet.",
        ],
        "monitoring_title": "How to monitor more closely",
        "monitoring": [
            "Check the prediction every morning, and again whenever new readings come in.",
            "Watch the probability chart. If the chance of Unhealthy keeps rising, get the full public advisory ready.",
            "Compare each prediction with the recorded category, since Moderate is the hardest category for the model.",
            "Record the advisories sent and the conditions in the monitoring log.",
        ],
        "escalate": "Move to the Unhealthy steps if the prediction becomes Unhealthy, or if conditions stay at Moderate for several days.",
    },
    "Unhealthy": {
        "headline": "Decision: Issue a public advisory and consider preventive measures.",
        "meaning": "Air quality is likely to be unhealthy for the general public, and sensitive groups are at the highest risk.",
        "now": [
            "Issue the public advisory through the LGU's official channels, such as its social media page, barangay announcements, and local radio.",
            "Advise residents to limit outdoor activity and wear a mask when going out, with extra guidance for children, the elderly, and people with respiratory conditions.",
            "Notify the city health office so clinics and health centers can prepare for more respiratory complaints.",
            "Coordinate with the traffic office and the DENR-EMB regional office on preventive measures, such as traffic measures or temporary restrictions, following local rules.",
        ],
        "monitoring_title": "How to monitor during an advisory",
        "monitoring": [
            "Check the prediction and the recorded readings every day until the city is back to Moderate or Good.",
            "Keep a record of the advisories released and the measures taken.",
            "Update the public when conditions improve.",
        ],
        "escalate": "Lower the alert level once the prediction returns to Moderate or Good and stays there for several days.",
    },
}

MODEL_NOTES = [
    ("Decision Tree (baseline)", "A single tree that splits the data using simple yes/no rules. Easy to explain, but it can be unstable."),
    ("Bagging", "Trains many trees on different random samples of the data and combines their votes, which makes predictions more stable."),
    ("Random Forest", "Like bagging, but each tree also uses a random subset of the features, so the trees are less alike."),
    ("Gradient Boosting", "Builds trees one after another, and each new tree tries to fix the mistakes of the previous ones."),
    ("AdaBoost", "Also builds simple learners in sequence, giving more weight to the cases that earlier learners got wrong."),
]

def categorize_aqi(aqi):
    r = round(aqi)
    if r <= 2:
        return "Good"
    elif r == 3:
        return "Moderate"
    return "Unhealthy"


def get_season(month):
    return "wet" if month in [6, 7, 8, 9, 10, 11] else "dry"

def render_recommendation(prediction, probs, classes):
    info = ACTIONS[prediction]
    box = {"Good": st.success, "Moderate": st.warning, "Unhealthy": st.error}[prediction]
    box(f"**{info['headline']}**\n\n{info['meaning']}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### What to do now")
        for step in info["now"]:
            st.markdown(f"- {step}")
    with col2:
        st.markdown(f"#### {info['monitoring_title']}")
        for step in info["monitoring"]:
            st.markdown(f"- {step}")

    st.markdown("#### When to change the alert level")
    st.write(info["escalate"])

    top_two = sorted(zip(classes, probs), key=lambda x: -x[1])[:2]
    if top_two[0][1] - top_two[1][1] < 0.15:
        st.info(
            f"This is a close call between {top_two[0][0]} and {top_two[1][0]}. "
            "Officers should review the probabilities before deciding."
        )
    st.caption(
        "These are suggested actions for the LGU to consider. They are not official DENR-EMB "
        "or health department protocols, and the final decision should always involve human judgment."
    )

@st.cache_resource
def load_model(name):
    return joblib.load(f"{ASSETS}/models/{MODEL_FILES[name]}")


@st.cache_data
def load_data():
    return pd.read_csv(f"{ASSETS}/app_data.csv", parse_dates=["date"])


@st.cache_data
def load_results():
    test = pd.read_csv(f"{ASSETS}/test_results.csv")
    train = pd.read_csv(f"{ASSETS}/train_results.csv")
    cost = pd.read_csv(f"{ASSETS}/error_costs.csv")
    with open(f"{ASSETS}/confusion_matrices.json") as f:
        cms = json.load(f)
    return test, train, cost, cms


data = load_data()
test_res, train_res, cost_res, cms = load_results()
all_cities = sorted(data["city_name"].unique())
n_train = int((data["split"] == "train").sum())
n_test = int((data["split"] == "test").sum())


def predict(model_name, features_df):
    model = load_model(model_name)
    probs = model.predict_proba(features_df[FEATURES])[0]
    classes = list(model.classes_)
    return classes[int(probs.argmax())], probs, classes


def render_prediction_insights(prediction, probs, classes, model_name):
    st.markdown(f"### Predicted category: **{prediction}**")
    st.caption(f"Model used: {model_name}" + (" (recommended)" if model_name == RECOMMENDED else ""))
    st.write("Probability for each category:")
    st.bar_chart(pd.DataFrame({"Probability": probs}, index=classes))
    st.markdown("#### What does this prediction mean?")
    st.write(CATEGORY_MEANING[prediction])
    st.write(f"The model gives this category a probability of {probs.max():.0%}.")
    m = test_res[test_res["Model"] == model_name].iloc[0]
    st.caption(
        f"On the test months (Sep–Dec 2024), this model caught {m['Moderate recall']:.0%} of the actual "
        f"Moderate days and {m['Unhealthy recall']:.0%} of the actual Unhealthy days."
    )
    with st.expander("Category guide"):
        for text in CATEGORY_MEANING.values():
            st.write(f"- {text}")


# ---------- Sidebar ----------
st.sidebar.header("Select Model, City, and Date")
labels = [f"{m} (recommended)" if m == RECOMMENDED else m for m in MODEL_FILES]
model_name = st.sidebar.selectbox("Model", labels, key="sb_model").replace(" (recommended)", "")
city = st.sidebar.selectbox("City", all_cities, key="sb_city")
city_df = data[data["city_name"] == city].sort_values("date")
city_test = city_df[city_df["split"] == "test"]
min_d, max_d = city_test["date"].min().date(), city_test["date"].max().date()
selected_date = st.sidebar.date_input(
    "Date (test months only)", value=max_d, min_value=min_d, max_value=max_d, key="sb_date"
)
st.sidebar.caption("Dates are limited to Sep–Dec 2024, which the models did not see during training.")

row = city_test[city_test["date"] == pd.Timestamp(selected_date)]
prediction = probs = classes = None
if not row.empty:
    prediction, probs, classes = predict(model_name, row)

st.title("Air Quality Advisory Support for Philippine Cities")
tabs = st.tabs([
    "Overview", "Data Overview", "Models & Evaluation", "ML Forecast",
    "Business Recommendation", "Test Your Own Data", "Responsible Use",
])

# ---------- Overview ----------
with tabs[0]:
    st.subheader("Project Description")
    st.write(
        "This app is a decision-support tool for LGU environmental offices and DENR-EMB regional offices. "
        "It predicts whether a city's air quality on a given day will be Good, Moderate, or Unhealthy, and "
        "suggests what the LGU can do about it. You can choose among five trained models. Bagging is the "
        "recommended one. This project supports SDG 11 (Sustainable Cities and Communities)."
    )
    st.subheader("Statement of the Problem")
    st.write(
        "LGUs and environmental agencies mostly rely on real-time or past air quality readings, so they "
        "respond only after pollution has become unhealthy. It is hard for them to anticipate air quality "
        "and prepare advisories or preventive measures ahead of time."
    )
    st.subheader("General Objective")
    st.write(
        "To develop a machine learning-based decision-support tool that predicts the air quality category "
        "of Philippine cities, so LGUs and environmental agencies can prepare advisories and preventive "
        "measures earlier."
    )
    st.subheader("Specific Objectives")
    st.markdown(
        "1. Describe air quality patterns across 138 Philippine cities.\n"
        "2. Prepare the data and build lag, rolling-average, and seasonal features without data leakage.\n"
        "3. Compare a baseline Decision Tree with four ensemble models using a date-based evaluation.\n"
        "4. Recommend a model using performance, error costs, and stakeholder needs.\n"
        "5. Turn each prediction into a recommended action for LGU officers."
    )
    st.subheader("Users and Decision Supported")
    st.write(
        "Primary users: LGU environmental offices or DENR-EMB regional offices. Decision supported: whether "
        "to issue an early advisory or take preventive action. The public benefits indirectly."
    )

# ---------- Data Overview ----------
with tabs[1]:
    st.subheader("Dataset")
    st.write(
        "Source: Philippine Cities Air Quality Index Data 2024 (Kaggle, uploaded by bwandowando), CC0 Public "
        "Domain. It covers 138 Philippine cities from November 2023 to December 2024, with readings about "
        "every 3 hours (455,854 readings in 14 monthly files)."
    )
    st.write(
        f"After averaging the readings per city per day and removing each city's first days (too little "
        f"history for the lag features), {len(data):,} city-day rows remain."
    )
    st.subheader("Target")
    st.write("The target is the day's air quality category from the 1–5 AQI scale: Good (1–2), Moderate (3), Unhealthy (4–5).")
    share = data["aqi_category"].value_counts(normalize=True).reindex(CLASS_ORDER)
    st.bar_chart(data["aqi_category"].value_counts().reindex(CLASS_ORDER))
    st.caption(
        f"The classes are very imbalanced: Good {share['Good']:.1%}, Moderate {share['Moderate']:.1%}, "
        f"Unhealthy {share['Unhealthy']:.1%}. Accuracy alone would be misleading, so each category is checked separately."
    )
    st.subheader("Features Used by the Model")
    st.dataframe(pd.DataFrame([
        ("prev_day_aqi", "Yesterday's average AQI (1–5)"),
        ("prev_day_category", "Yesterday's category (Good, Moderate, Unhealthy)"),
        ("aqi_7day_avg", "Average AQI over the previous 7 days"),
        ("city_name, coord.lat, coord.lon", "The city and its location"),
        ("month, day_of_week, season", "Calendar context (wet season: June–November, dry season: December–May)"),
    ], columns=["Feature", "Meaning"]), hide_index=True)
    st.subheader("Data Preparation")
    st.markdown(
        "- The 14 monthly files were merged, and readings were averaged per city per day.\n"
        "- The AQI was grouped into three categories to make the target.\n"
        "- Same-day pollutant readings were not used as inputs, because the AQI is calculated from them and "
        "they would not be known when the prediction is needed.\n"
        "- Lag and rolling features use past days only.\n"
        "- Encoding is done inside a scikit-learn Pipeline, so live inputs get the same preprocessing as the training data.\n"
        f"- Date-based split: models were trained on Jan–Aug 2024 ({n_train:,} rows) and tested on Sep–Dec 2024 "
        f"({n_test:,} rows). Nov–Dec 2023 rows are not used in this evaluation."
    )
    st.subheader(f"Air Quality History: {city}")
    st.line_chart(city_df.set_index("date")[["main.aqi", "aqi_7day_avg"]])
    st.caption("Daily average AQI (1–5 scale) and the 7-day rolling average.")

# ---------- Models & Evaluation ----------
with tabs[2]:
    st.subheader("Model Comparison")
    st.write(
        f"Five models were trained on {n_train:,} days (Jan–Aug 2024) and tested on {n_test:,} days "
        "(Sep–Dec 2024) that they had not seen."
    )
    st.dataframe(test_res.round(3), hide_index=True)
    st.markdown("**Per-class recall** (the share of actual days of each category that the model caught)")
    st.bar_chart(test_res.set_index("Model")[["Good recall", "Moderate recall", "Unhealthy recall"]])
    st.markdown(f"**Confusion matrix: {model_name}** (change the model in the sidebar)")
    cm = pd.DataFrame(
        cms[model_name],
        index=[f"Actual {c}" for c in CLASS_ORDER],
        columns=[f"Predicted {c}" for c in CLASS_ORDER],
    )
    c1, c2 = st.columns(2)
    c1.write("Counts")
    c1.dataframe(cm)
    c2.write("% of each actual class")
    c2.dataframe((cm.div(cm.sum(axis=1), axis=0) * 100).round(0))
    st.markdown("**Overfitting check (training months vs. test months)**")
    tt = train_res.merge(test_res, on="Model", suffixes=(" (train)", " (test)"))
    st.dataframe(tt[[
        "Model", "F1 (macro) (train)", "F1 (macro) (test)", "Moderate recall (train)",
        "Moderate recall (test)", "Unhealthy recall (train)", "Unhealthy recall (test)",
    ]].round(3), hide_index=True)
    st.markdown("**Estimated error cost (hypothetical)**")
    st.dataframe(cost_res, hide_index=True)
    st.caption(
        "Assumed costs: missed Unhealthy = 5,000, missed Moderate = 1,000, false alarm = 150 units. "
        "These are classroom assumptions, not real LGU figures."
    )
    with st.expander("What each model does"):
        for name, text in MODEL_NOTES:
            st.markdown(f"- **{name}:** {text}")
    b = test_res.set_index("Model")
    c = cost_res.set_index("Model")
    st.success(
        f"**Recommended model: {RECOMMENDED}.** On the test months it caught "
        f"{b.loc[RECOMMENDED, 'Unhealthy recall']:.0%} of Unhealthy days and {b.loc[RECOMMENDED, 'Moderate recall']:.0%} "
        f"of Moderate days, with an estimated error cost of {c.loc[RECOMMENDED, 'Estimated Error Cost']:,.0f}. "
        f"Random Forest caught {b.loc['Random Forest', 'Unhealthy recall']:.0%} of Unhealthy days at a cost of "
        f"{c.loc['Random Forest', 'Estimated Error Cost']:,.0f}. AdaBoost has the lowest cost but never predicts "
        "Moderate, and Gradient Boosting misses many Moderate and Unhealthy days. Compared with the single "
        f"Decision Tree, Bagging has a slightly better F1 ({b.loc[RECOMMENDED, 'F1 (macro)']:.3f} vs. "
        f"{b.loc['Decision Tree', 'F1 (macro)']:.3f}) at a similar cost, so the improvement is modest. "
        "Bagging was also the most consistent model across the two date-based experiments."
    )

# ---------- ML Forecast ----------
with tabs[3]:
    st.subheader(f"Prediction for {city} on {selected_date}")
    if row.empty:
        st.warning("No data is available for this city on this date. Try another date.")
    else:
        r = row.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Previous day AQI", f"{r['prev_day_aqi']:.2f}")
        c2.metric("7-day average AQI", f"{r['aqi_7day_avg']:.2f}")
        c3.metric("Previous day category", r["prev_day_category"])

        st.markdown("#### Features used")
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            with st.container(border=True):
                st.markdown("**Recent history**")
                st.caption("Shows how air quality has been trending.")
                st.write(f"Previous day AQI: {r['prev_day_aqi']:.2f}")
                st.write(f"Previous day category: {r['prev_day_category']}")
                st.write(f"7-day average AQI: {r['aqi_7day_avg']:.2f}")
        with f2:
            with st.container(border=True):
                st.markdown("**Location**")
                st.caption("Pollution risk differs from area to area.")
                st.write(f"City: {r['city_name']}")
                st.write(f"Latitude: {r['coord.lat']:.4f}")
                st.write(f"Longitude: {r['coord.lon']:.4f}")
        with f3:
            with st.container(border=True):
                st.markdown("**Calendar**")
                st.caption("Season and month can affect air quality patterns.")
                st.write(f"Month: {int(r['month'])}")
                st.write(f"Day of week (0 = Monday): {int(r['day_of_week'])}")
                st.write(f"Season: {r['season']}")
        with f4:
            with st.container(border=True):
                st.markdown("**Context only (not used by the model)**")
                st.caption("Daily average pollutant readings for this date, shown for reference.")
                for col, label in POLLUTANTS.items():
                    if col in row.columns:
                        st.write(f"{label}: {r[col]:.2f}")
        st.caption(
            "The pollutant readings are not model inputs. The AQI is calculated from them, so using them "
            "to predict the AQI would be circular."
        )

        render_prediction_insights(prediction, probs, classes, model_name)

        st.write(f"Actual recorded category for this date: **{r['aqi_category']}**")
        if r["aqi_category"] == prediction:
            st.success("The prediction matches the recorded category.")
        else:
            st.info("The prediction differs from the recorded category.")
        st.caption("See the Models & Evaluation tab for how the models compare.")

# ---------- Business Recommendation ----------
with tabs[4]:
    st.subheader("Recommended Action")
    if prediction is None:
        st.warning("Select a city and date that has data to see a recommendation.")
    else:
        st.write(
            f"Based on the {model_name} model's prediction for {city} on {selected_date}: **{prediction}**. "
            "The LGU needs enough warning to prepare advisories or preventive measures, so the steps below "
            "show what to do now and how to keep monitoring."
        )
        render_recommendation(prediction, probs, classes)

# ---------- Test Your Own Data ----------
with tabs[5]:
    st.subheader("Test Your Own Data")
    st.write(
        f"Enter your own values for the model's inputs. The selected model ({model_name}) will make the prediction. "
        "The other features (location, month, day of week, season, and the previous day's category) are worked out automatically."
    )
    col_a, col_b = st.columns(2)
    with col_a:
        t_city = st.selectbox("City", all_cities, key="t_city")
        t_date = st.date_input("Date to predict", value=max_d, key="t_date")
    with col_b:
        t_prev = st.number_input("Previous day AQI (1.0 to 5.0)", min_value=1.0, max_value=5.0, value=2.0, step=0.1, key="t_prev")
        t_avg = st.number_input("7-day average AQI (1.0 to 5.0)", min_value=1.0, max_value=5.0, value=2.0, step=0.1, key="t_avg")

    city_row = data[data["city_name"] == t_city].iloc[0]
    t_ts = pd.Timestamp(t_date)
    manual = pd.DataFrame([{
        "city_name": t_city,
        "coord.lat": city_row["coord.lat"],
        "coord.lon": city_row["coord.lon"],
        "month": t_ts.month,
        "day_of_week": t_ts.dayofweek,
        "season": get_season(t_ts.month),
        "prev_day_aqi": t_prev,
        "prev_day_category": categorize_aqi(t_prev),
        "aqi_7day_avg": t_avg,
    }])
    with st.expander("Inputs sent to the model"):
        st.dataframe(manual, hide_index=True)

    m_pred, m_probs, m_classes = predict(model_name, manual)
    render_prediction_insights(m_pred, m_probs, m_classes, model_name)
    st.markdown("---")
    st.subheader("Recommended Action")
    render_recommendation(m_pred, m_probs, m_classes)

# ---------- Responsible Use ----------
with tabs[6]:
    st.subheader("Responsible Use, Limitations, and Assumptions")
    mod = test_res.set_index("Model").loc[RECOMMENDED, "Moderate recall"]
    st.markdown(
        "- **Decision support only.** The result is an estimate based on past patterns. The final decision to issue "
        "an advisory should always involve human judgment from the LGU or environmental officer.\n"
        f"- **Moderate is the hardest category.** The recommended model caught {mod:.0%} of the actual Moderate days "
        "on the test months, so close calls should be reviewed before acting.\n"
        "- **Short-term and historical.** The model depends mostly on yesterday's AQI and the 7-day average, so it "
        "works as a next-day estimate and not a multi-day forecast. This demo uses the dataset's historical dates, not live readings.\n"
        "- **Limited data.** The dataset covers about 14 months, so patterns across several years are not covered. "
        "Cities with less data may have less reliable predictions.\n"
        "- **Location matters.** Location is one of the model's inputs, so predictions follow past patterns for each place "
        "and may not reflect sudden local changes.\n"
        "- **Privacy.** The app uses city-level air quality data only. No personal data is collected or stored.\n"
        "- **Costs and actions are assumptions.** The error costs are hypothetical, and the recommended actions are "
        "suggestions for the LGU to consider, not official DENR-EMB or health department protocols. They should be "
        "checked with real LGU and DENR-EMB stakeholders before use."
    )