import re
import pandas as pd
import streamlit as st


def _inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {
            --bg-0: #f7f5f2;
            --bg-1: #ebe5dc;
            --ink-0: #1f2022;
            --ink-1: #4f545a;
            --accent: #0d9488;
            --accent-soft: #c7f4ef;
            --card: #fffdf9;
            --line: #d9d2c7;
            --ok: #2f855a;
        }

        html, body, [class*="css"] {
            font-family: 'Space Grotesk', sans-serif;
            color: var(--ink-0);
        }

        .stApp {
            background:
                radial-gradient(1200px 400px at 10% -10%, #fff8e6 0%, transparent 60%),
                radial-gradient(1000px 420px at 100% 0%, #d7efe8 0%, transparent 58%),
                linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 100%);
        }

        .hero-wrap {
            border: 1px solid var(--line);
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(247,250,249,0.8) 100%);
            padding: 1.25rem 1.4rem;
            margin-bottom: 1rem;
            box-shadow: 0 10px 25px rgba(31, 32, 34, 0.06);
        }

        .hero-title {
            font-size: clamp(1.5rem, 2.6vw, 2.3rem);
            letter-spacing: -0.02em;
            margin: 0;
            color: var(--ink-0);
        }

        .hero-sub {
            margin: 0.35rem 0 0 0;
            color: var(--ink-1);
            font-size: 0.98rem;
        }

        .section-card {
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 0.7rem 1rem 0.95rem 1rem;
            background: var(--card);
            margin-top: 0.45rem;
            box-shadow: 0 7px 18px rgba(31, 32, 34, 0.05);
        }

        .result-chip {
            display: inline-block;
            background: var(--accent-soft);
            color: #0f5f58;
            border: 1px solid #96dbd2;
            border-radius: 999px;
            padding: 0.28rem 0.68rem;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.85rem;
            margin-bottom: 0.25rem;
        }

        .result-text {
            margin: 0.2rem 0 0.2rem 0;
            font-size: 1.3rem;
            font-weight: 650;
            color: var(--ok);
        }

        [data-testid="stForm"] {
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 0.9rem 1rem 0.6rem 1rem;
            background: var(--card);
        }

        [data-testid="stMetricValue"] {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 1.05rem;
        }

        .stButton > button, .stFormSubmitButton > button {
            border-radius: 12px;
            border: 1px solid #0f766e;
            background: linear-gradient(180deg, #14b8a6 0%, #0f766e 100%);
            color: #f7fffe;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_field(col, values, category_options, default_values):
    default_value = default_values.get(col)

    if col in category_options:
        options = category_options[col] or ["Unknown"]
        default_str = str(default_value) if default_value is not None else options[0]
        default_index = options.index(default_str) if default_str in options else 0
        values[col] = st.selectbox(col, options=options, index=default_index)
        return

    # Heuristic: decide whether to show a numeric input or a text input
    def _is_numeric_column(name: str) -> bool:
        name_l = name.lower()
        numeric_tokens = ("age", "population", "area", "density", "num", "count", "-2020")
        return any(tok in name_l for tok in numeric_tokens)

    if default_value is not None:
        try:
            numeric_default = float(default_value)
            values[col] = st.number_input(col, value=numeric_default)
            return
        except Exception:
            pass

    if _is_numeric_column(col):
        # default to 0 for numeric-looking columns
        values[col] = st.number_input(col, value=0.0)
    else:
        # free-form text for categorical or unknown columns
        values[col] = st.text_input(col, value=str(default_value or ""))


def _build_input_row(feature_columns, category_options, default_values):
    values = {}

    # Render a single free-text area for the model's text column (prefer 'clean_text')
    if "clean_text" in feature_columns or "text" in feature_columns:
        input_key = "clean_text" if "clean_text" in feature_columns else "text"
        default_text = default_values.get(input_key) or default_values.get("text") or ""
        values[input_key] = st.text_area(
            "Tweet Text",
            value=str(default_text),
            height=140,
            help="Enter tweet text to classify sentiment.",
            placeholder="Write a tweet message here...",
        )

    remaining_cols = [col for col in feature_columns if col not in ("text", "clean_text")]
    if not remaining_cols:
        return values

    st.markdown("### Metadata")
    left_col, right_col = st.columns(2)
    for idx, col in enumerate(remaining_cols):
        with left_col if idx % 2 == 0 else right_col:
            _render_field(col, values, category_options, default_values)

    return values


def clean_tweet_text(text):
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)

        text = str(text).lower()
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"[^a-z\s]", " ", text)

        stop_words = set(stopwords.words('english'))
        tokens = [word for word in text.split() if word not in stop_words]

        lemmatizer = WordNetLemmatizer()
        clean_tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return " ".join(clean_tokens)
    except Exception:
        # Lightweight fallback cleaning if nltk resources aren't available
        text = str(text).lower()
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        return " ".join(text.split())


def render_app(bundle):
    st.set_page_config(page_title="Tweet Sentiment Predictor", layout="wide")
    _inject_styles()

    st.markdown(
        """
        <div class="hero-wrap">
            <h1 class="hero-title">Tweet Sentiment Predictor</h1>
            <p class="hero-sub">Logistic Regression model trained on tweet text and contextual metadata.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pipeline = bundle["pipeline"]
    feature_columns = bundle["feature_columns"]
    class_labels = bundle.get("class_labels", [])
    category_options = bundle.get("category_options", {})
    default_values = bundle.get("default_values", {})

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    with st.form("prediction_form"):
        input_values = _build_input_row(feature_columns, category_options, default_values)
        submitted = st.form_submit_button("Predict Sentiment")
    st.markdown('</div>', unsafe_allow_html=True)

    if not submitted:
        st.info("Fill in the inputs and click Predict Sentiment.")
        return

    input_df = pd.DataFrame([input_values])
    # Ensure the pipeline receives the same 'clean_text' column used at training
    if "text" in input_df.columns and "clean_text" not in input_df.columns:
        input_df["clean_text"] = input_df["text"].apply(clean_tweet_text)
    # If user provided a 'clean_text' field directly, normalize it as well
    if "clean_text" in input_df.columns:
        input_df["clean_text"] = input_df["clean_text"].astype(str).apply(clean_tweet_text)

    # Some ColumnTransformer pipelines expect specific input columns. If any are missing,
    # try to infer them from the fitted preprocessor and add reasonable defaults.
    try:
        preprocessor = pipeline.named_steps.get("preprocessor")
        required_cols = []
        col_types = {}

        if preprocessor is not None:
            # If fitted, scikit-learn may expose 'feature_names_in_' on the preprocessor
            if hasattr(preprocessor, "feature_names_in_"):
                required_cols = list(preprocessor.feature_names_in_)
                # best-effort: mark any 'clean' containing column as text
                for c in required_cols:
                    if "clean" in c or "text" in c:
                        col_types[c] = "text"
            elif hasattr(preprocessor, "transformers_"):
                # inspect each transformer to figure out which columns are text/categorical/numeric
                for name, transformer, cols_spec in preprocessor.transformers_:
                    if cols_spec is None:
                        continue
                    cols = list(cols_spec) if isinstance(cols_spec, (list, tuple)) else [cols_spec]
                    t_type = "unknown"
                    # detect transformer type
                    try:
                        # OneHotEncoder -> categorical
                        import sklearn
                        from sklearn.preprocessing import OneHotEncoder, StandardScaler
                        from sklearn.feature_extraction.text import TfidfVectorizer
                        if isinstance(transformer, TfidfVectorizer) or (hasattr(transformer, "__class__") and transformer.__class__.__name__ == "TfidfVectorizer"):
                            t_type = "text"
                        else:
                            # If transformer is a Pipeline, inspect its steps
                            if hasattr(transformer, "steps"):
                                step_names = [s[0] for s in transformer.steps]
                                if any("onehot" in s or "one_hot" in s or "onehotencoder" in s.lower() for s in step_names):
                                    t_type = "categorical"
                                elif any("scaler" in s or "imputer" in s for s in step_names):
                                    t_type = "numeric"
                            else:
                                # direct estimator checks
                                from sklearn.preprocessing import OneHotEncoder
                                if isinstance(transformer, OneHotEncoder):
                                    t_type = "categorical"
                    except Exception:
                        t_type = "unknown"

                    for c in cols:
                        required_cols.append(c)
                        col_types[c] = t_type

        # Fill missing required columns with sensible defaults based on detected types
        if required_cols:
            missing = [c for c in required_cols if c not in input_df.columns]
            for c in missing:
                ctype = col_types.get(c, "unknown")
                if ctype == "text" or "clean" in c or "text" in c:
                    input_df[c] = input_df.get("clean_text") or input_df.get("text") or ""
                elif ctype == "numeric":
                    input_df[c] = 0
                elif ctype == "categorical":
                    # prefer any provided category options for sensible default
                    options = category_options.get(c) if isinstance(category_options, dict) else None
                    if options:
                        input_df[c] = options[0]
                    else:
                        input_df[c] = ""
                else:
                    # conservative fallback
                    input_df[c] = input_df.get("clean_text") or input_df.get("text") or ""
    except Exception:
        # If anything goes wrong while inferring columns, proceed and let sklearn raise the original error.
        pass
    prediction = pipeline.predict(input_df)[0]

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<span class="result-chip">PREDICTION RESULT</span>', unsafe_allow_html=True)
    st.markdown(f'<p class="result-text">{prediction}</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if hasattr(pipeline, "predict_proba") and class_labels:
        proba = pipeline.predict_proba(input_df)[0]
        proba_df = pd.DataFrame(
            {"sentiment": class_labels, "probability": proba}
        ).sort_values("probability", ascending=False)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Class Probabilities")

        top_k = min(3, len(proba_df))
        if top_k > 0:
            metric_cols = st.columns(top_k)
            for idx in range(top_k):
                row = proba_df.iloc[idx]
                metric_cols[idx].metric(row["sentiment"], f"{row['probability']:.2%}")

        st.dataframe(proba_df, use_container_width=True)
        st.bar_chart(proba_df.set_index("sentiment"))
        st.markdown('</div>', unsafe_allow_html=True)
