from pathlib import Path

import joblib
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from ui.app import render_app


ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "logistic_sentiment_pipeline.joblib"


def load_bundle(model_path: Path):
	if not model_path.exists():
		st.error(
			f"Model file not found at {model_path}. Run notebooks/train.ipynb first to train and export the model."
		)
		st.stop()
	loaded = joblib.load(model_path)

	# If a bundle dict was saved, return it directly
	if isinstance(loaded, dict):
		return loaded

	# If a plain sklearn Pipeline was saved, wrap it into the expected bundle shape
	if isinstance(loaded, Pipeline):
		pipeline = loaded
		# Try to infer feature column names from the ColumnTransformer used in training
		feature_columns = ["clean_text"]
		try:
			preprocessor = pipeline.named_steps.get("preprocessor")
			if isinstance(preprocessor, ColumnTransformer):
				cols = []
				for name, transformer, cols_spec in preprocessor.transformers_:
					if cols_spec is None:
						continue
					if isinstance(cols_spec, (list, tuple)):
						cols.extend(list(cols_spec))
					else:
						cols.append(cols_spec)
				# keep the original column names as used during training (e.g. 'clean_text')
				feature_columns = cols or feature_columns
		except Exception:
			feature_columns = ["clean_text"]

		class_labels = []
		try:
			classifier = pipeline.named_steps.get("classifier")
			if hasattr(classifier, "classes_"):
				class_labels = [str(c) for c in classifier.classes_]
		except Exception:
			pass

		bundle = {
			"pipeline": pipeline,
			"feature_columns": feature_columns,
			"class_labels": class_labels,
			"category_options": {},
			# provide an explicit default for the cleaned text field
			"default_values": {"clean_text": ""},
		}

		return bundle

	# Fallback: return what was loaded
	return loaded


def main():
	bundle = load_bundle(MODEL_PATH)
	render_app(bundle)


if __name__ == "__main__":
	main()
