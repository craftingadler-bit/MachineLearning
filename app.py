from __future__ import annotations

import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.joblib"

# Replace these values with the thresholds selected from Data_1 and Data_2.
# They must remain fixed when the application is used for Data_3.
CONFIDENCE_THRESHOLD = 0.65
MARGIN_THRESHOLD = 0.15

# Used only when the saved model predicts numeric labels.
CATEGORY_NAMES = {
    "1": "Non-current assets",
    "2": "Current assets",
    "3": "Equity",
    "4": "Liabilities",
    "5": "Income / Revenue",
    "6": "Expenses",
    "7": "Financial result / Taxes",
}

CATEGORY_ORDER = list(CATEGORY_NAMES.values())

RESULT_COLUMNS = [
    "recommended_category",
    "confidence",
    "second_category",
    "margin",
    "review_required",
    "review_reason",
    "expected_category_normalized",
    "prediction_correct",
]


@st.cache_resource
def load_model(model_path: Path):
    """Load the trusted, fitted scikit-learn pipeline once."""
    return joblib.load(model_path)


def category_name(label: object) -> str:
    """Convert numeric model labels to readable category names."""
    label_text = str(label)
    return CATEGORY_NAMES.get(label_text, label_text)


def normalize_category_label(label: object) -> str:
    """Normalize numeric and textual target labels from the uploaded file."""
    if pd.isna(label):
        return ""

    label_text = str(label).strip()
    if not label_text:
        return ""

    if label_text in CATEGORY_NAMES:
        return CATEGORY_NAMES[label_text]

    # Excel may read numeric labels as values such as 1.0 instead of 1.
    try:
        numeric_label = float(label_text)
        if numeric_label.is_integer():
            numeric_key = str(int(numeric_label))
            if numeric_key in CATEGORY_NAMES:
                return CATEGORY_NAMES[numeric_key]
    except ValueError:
        pass

    canonical_names = {
        category.lower(): category for category in CATEGORY_ORDER
    }
    return canonical_names.get(label_text.lower(), label_text)


def classify_descriptions(
    data: pd.DataFrame,
    description_column: str,
    model,
) -> pd.DataFrame:
    """Add category recommendations while preserving all original columns."""
    result = data.copy()

    # Avoid silently overwriting results from an earlier application run.
    result = result.drop(
        columns=[column for column in RESULT_COLUMNS if column in result.columns],
        errors="ignore",
    )

    descriptions = (
        result[description_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    valid_rows = descriptions.ne("")

    result["recommended_category"] = pd.Series(
        "", index=result.index, dtype="object"
    )
    result["confidence"] = np.nan
    result["second_category"] = pd.Series(
        "", index=result.index, dtype="object"
    )
    result["margin"] = np.nan
    result["review_required"] = True
    result["review_reason"] = pd.Series(
        "Missing description", index=result.index, dtype="object"
    )

    if not valid_rows.any():
        return result

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "The saved model does not provide predict_proba(). "
            "Confidence and margin cannot be calculated."
        )

    probabilities = model.predict_proba(descriptions.loc[valid_rows])
    classes = getattr(model, "classes_", None)

    if classes is None:
        raise TypeError("The saved model does not provide fitted class labels.")
    if probabilities.shape[1] < 2:
        raise ValueError("At least two model classes are required.")

    classes = np.asarray(classes)
    ranking = np.argsort(probabilities, axis=1)
    best_indices = ranking[:, -1]
    second_indices = ranking[:, -2]
    row_indices = np.arange(len(probabilities))

    best_probabilities = probabilities[row_indices, best_indices]
    second_probabilities = probabilities[row_indices, second_indices]
    margins = best_probabilities - second_probabilities

    low_confidence = best_probabilities < CONFIDENCE_THRESHOLD
    small_margin = margins < MARGIN_THRESHOLD
    review_required = low_confidence | small_margin

    review_reasons = np.select(
        [
            low_confidence & small_margin,
            low_confidence,
            small_margin,
        ],
        [
            "Low confidence and small margin",
            "Low confidence",
            "Small margin",
        ],
        default="No review required",
    )

    best_labels = [category_name(classes[index]) for index in best_indices]
    second_labels = [category_name(classes[index]) for index in second_indices]

    result.loc[valid_rows, "recommended_category"] = best_labels
    result.loc[valid_rows, "confidence"] = best_probabilities
    result.loc[valid_rows, "second_category"] = second_labels
    result.loc[valid_rows, "margin"] = margins
    result.loc[valid_rows, "review_required"] = review_required
    result.loc[valid_rows, "review_reason"] = review_reasons

    return result


def evaluate_predictions(
    result: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, dict]:
    """Compare recommendations with a selected ground-truth column."""
    evaluated_result = result.copy()
    normalized_targets = evaluated_result[target_column].map(
        normalize_category_label
    )
    predictions = (
        evaluated_result["recommended_category"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    known_target = normalized_targets.isin(CATEGORY_ORDER)
    valid_prediction = predictions.isin(CATEGORY_ORDER)
    evaluation_rows = known_target & valid_prediction

    evaluated_result["expected_category_normalized"] = pd.Series(
        normalized_targets, index=evaluated_result.index, dtype="object"
    )
    evaluated_result["prediction_correct"] = pd.Series(
        pd.NA, index=evaluated_result.index, dtype="boolean"
    )
    evaluated_result.loc[evaluation_rows, "prediction_correct"] = (
        predictions.loc[evaluation_rows]
        == normalized_targets.loc[evaluation_rows]
    )

    if not evaluation_rows.any():
        raise ValueError(
            "No rows contain both a valid description prediction and a known "
            "target category. The target column must use labels 1-7 or the "
            "seven category names."
        )

    y_true = normalized_targets.loc[evaluation_rows]
    y_pred = predictions.loc[evaluation_rows]
    observed_categories = set(y_true) | set(y_pred)
    active_labels = [
        category for category in CATEGORY_ORDER
        if category in observed_categories
    ]

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=active_labels,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=active_labels,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=active_labels,
        output_dict=True,
        zero_division=0,
    )
    report_table = pd.DataFrame(report).transpose().round(4)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=active_labels,
    )
    matrix_table = pd.DataFrame(
        matrix,
        index=[f"Actual: {category}" for category in active_labels],
        columns=[f"Predicted: {category}" for category in active_labels],
    )

    nonempty_targets = normalized_targets.ne("")
    unknown_target_rows = nonempty_targets & ~known_target
    missing_target_rows = ~nonempty_targets

    evaluation = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "evaluated_rows": int(evaluation_rows.sum()),
        "unknown_target_rows": int(unknown_target_rows.sum()),
        "missing_target_rows": int(missing_target_rows.sum()),
        "report": report_table,
        "confusion_matrix": matrix_table,
        "confusion_matrix_values": matrix,
        "active_labels": active_labels,
    }

    return evaluated_result, evaluation


def create_confusion_matrix_figure(evaluation: dict) -> go.Figure:
    """Create an interactive heatmap for the confusion matrix."""
    matrix = evaluation["confusion_matrix_values"]
    labels = evaluation["active_labels"]

    heatmap = go.Heatmap(
        z=matrix,
        x=labels,
        y=labels,
        colorscale="Blues",
        colorbar={"title": "Accounts"},
        hovertemplate=(
            "Actual: %{y}<br>"
            "Predicted: %{x}<br>"
            "Accounts: %{z}<extra></extra>"
        ),
    )

    annotations = []
    maximum = matrix.max() if matrix.size else 0
    for row_index, actual_category in enumerate(labels):
        for column_index, predicted_category in enumerate(labels):
            value = int(matrix[row_index, column_index])
            text_color = "white" if maximum and value > maximum / 2 else "black"
            annotations.append(
                {
                    "x": predicted_category,
                    "y": actual_category,
                    "text": str(value),
                    "showarrow": False,
                    "font": {"color": text_color, "size": 13},
                }
            )

    figure = go.Figure(data=[heatmap])
    figure.update_layout(
        annotations=annotations,
        height=680,
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        xaxis={
            "title": "Predicted category",
            "tickangle": -30,
            "side": "bottom",
        },
        yaxis={
            "title": "Actual category",
            "autorange": "reversed",
        },
    )
    return figure


def create_excel(
    result: pd.DataFrame,
    source_sheet: str,
    evaluation: dict | None = None,
) -> bytes:
    """Create the downloadable Excel workbook in memory."""
    processed = int(result["recommended_category"].ne("").sum())
    review_cases = int(result["review_required"].sum())
    accepted_cases = int((~result["review_required"]).sum())

    summary = pd.DataFrame(
        {
            "Measure": [
                "Source worksheet",
                "Total rows",
                "Processed descriptions",
                "Accepted recommendations",
                "Cases for expert review",
                "Confidence threshold",
                "Margin threshold",
                "Output type",
            ],
            "Value": [
                source_sheet,
                len(result),
                processed,
                accepted_cases,
                review_cases,
                CONFIDENCE_THRESHOLD,
                MARGIN_THRESHOLD,
                "Category recommendation",
            ],
        }
    )

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="Classification Results", index=False)
        summary.to_excel(writer, sheet_name="Model Information", index=False)

        if evaluation is not None:
            evaluation_summary = pd.DataFrame(
                {
                    "Measure": [
                        "Accuracy",
                        "Macro F1",
                        "Weighted F1",
                        "Evaluated rows",
                        "Rows with unknown target labels",
                        "Rows with missing target labels",
                    ],
                    "Value": [
                        evaluation["accuracy"],
                        evaluation["macro_f1"],
                        evaluation["weighted_f1"],
                        evaluation["evaluated_rows"],
                        evaluation["unknown_target_rows"],
                        evaluation["missing_target_rows"],
                    ],
                }
            )
            evaluation_summary.to_excel(
                writer, sheet_name="Evaluation Summary", index=False
            )
            evaluation["report"].to_excel(
                writer, sheet_name="Classification Report", index=True
            )
            evaluation["confusion_matrix"].to_excel(
                writer, sheet_name="Confusion Matrix", index=True
            )

        result_sheet = writer.sheets["Classification Results"]
        result_sheet.freeze_panes = "A2"
        result_sheet.auto_filter.ref = result_sheet.dimensions

        information_sheet = writer.sheets["Model Information"]
        information_sheet.freeze_panes = "A2"

    output.seek(0)
    return output.getvalue()


def default_description_index(columns: list[str]) -> int:
    """Prefer the English description fields used in the study."""
    preferred_columns = [
        "description_en_long",
        "description_en_short",
        "description",
    ]
    normalized = {str(column).lower(): index for index, column in enumerate(columns)}

    for preferred in preferred_columns:
        if preferred in normalized:
            return normalized[preferred]
    return 0


def default_target_index(columns: list[str]) -> int:
    """Prefer common names for a ground-truth category column."""
    preferred_columns = [
        "expected_category",
        "true_category",
        "category",
        "category_label",
        "target",
    ]
    normalized = {str(column).lower(): index for index, column in enumerate(columns)}

    for preferred in preferred_columns:
        if preferred in normalized:
            return normalized[preferred]
    return 0


def main() -> None:
    st.set_page_config(
        page_title="G/L Account Classification",
        page_icon="📊",
        layout="wide",
    )

    st.title("G/L Account Classification Prototype")
    st.write(
        "Upload an Excel file to receive category recommendations for "
        "General Ledger account descriptions."
    )
    st.info(
        "The results support an initial review. They are not final account "
        "mappings or accounting decisions."
    )

    if not MODEL_PATH.exists():
        st.error(
            "The file 'model.joblib' was not found. Save the fitted pipeline "
            "in the same folder as app.py."
        )
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload Excel file",
        type=["xlsx"],
        help="The original columns remain in the downloaded result.",
    )

    if uploaded_file is None:
        st.stop()

    try:
        file_bytes = uploaded_file.getvalue()
        workbook = pd.ExcelFile(io.BytesIO(file_bytes))
    except Exception as error:
        st.error(f"The Excel file could not be opened: {error}")
        st.stop()

    selected_sheet = st.selectbox("Select worksheet", workbook.sheet_names)

    try:
        data = pd.read_excel(
            io.BytesIO(file_bytes),
            sheet_name=selected_sheet,
        )
    except Exception as error:
        st.error(f"The selected worksheet could not be read: {error}")
        st.stop()

    if data.empty:
        st.warning("The selected worksheet does not contain any rows.")
        st.stop()
    if len(data.columns) == 0:
        st.warning("The selected worksheet does not contain any columns.")
        st.stop()

    columns = list(data.columns)
    description_column = st.selectbox(
        "Select the description column",
        columns,
        index=default_description_index(columns),
    )

    contains_correct_results = st.checkbox(
        "The Excel file contains the correct categories",
        help=(
            "Activate this option to compare the recommendations with a "
            "ground-truth column in the selected worksheet."
        ),
    )

    target_column = None
    if contains_correct_results:
        target_column = st.selectbox(
            "Select the column containing the correct categories",
            columns,
            index=default_target_index(columns),
        )

        if target_column == description_column:
            st.warning(
                "The description column and target column should not be the same."
            )

    st.subheader("Uploaded data")
    st.caption(f"{len(data):,} rows in worksheet '{selected_sheet}'")
    st.dataframe(data.head(20), width="stretch")

    if not st.button("Create recommendations", type="primary"):
        st.stop()

    try:
        model = load_model(MODEL_PATH)
        result = classify_descriptions(data, description_column, model)
        evaluation = None
        if contains_correct_results:
            result, evaluation = evaluate_predictions(result, target_column)
    except Exception as error:
        st.error(f"The classification could not be completed: {error}")
        st.stop()

    processed = int(result["recommended_category"].ne("").sum())
    review_cases = int(result["review_required"].sum())
    accepted_cases = int((~result["review_required"]).sum())

    st.success(f"{processed:,} descriptions were processed.")

    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Processed", f"{processed:,}")
    metric_2.metric("Accepted", f"{accepted_cases:,}")
    metric_3.metric("Expert review", f"{review_cases:,}")

    st.subheader("Classification results")
    st.dataframe(result.head(100), width="stretch")

    if evaluation is not None:
        st.subheader("Accuracy analysis")

        evaluation_1, evaluation_2, evaluation_3, evaluation_4 = st.columns(4)
        evaluation_1.metric(
            "Accuracy", f"{evaluation['accuracy']:.1%}"
        )
        evaluation_2.metric(
            "Macro F1", f"{evaluation['macro_f1']:.3f}"
        )
        evaluation_3.metric(
            "Weighted F1", f"{evaluation['weighted_f1']:.3f}"
        )
        evaluation_4.metric(
            "Evaluated rows", f"{evaluation['evaluated_rows']:,}"
        )

        excluded_rows = (
            evaluation["unknown_target_rows"]
            + evaluation["missing_target_rows"]
        )
        if excluded_rows:
            st.warning(
                f"{excluded_rows:,} rows were excluded from the accuracy "
                "analysis: "
                f"{evaluation['unknown_target_rows']:,} unknown target labels "
                f"and {evaluation['missing_target_rows']:,} missing labels."
            )

        st.subheader("Confusion matrix")
        st.caption(
            "Rows show the correct category and columns show the prediction. "
            "Correct predictions are located on the diagonal."
        )
        confusion_figure = create_confusion_matrix_figure(evaluation)
        st.plotly_chart(
            confusion_figure,
            width="stretch",
            config={
                "displayModeBar": True,
                "displaylogo": False,
            },
        )

        with st.expander("Show numerical confusion matrix"):
            st.dataframe(
                evaluation["confusion_matrix"],
                width="stretch",
            )

        st.subheader("Class-level performance")
        st.dataframe(evaluation["report"], width="stretch")

    excel_file = create_excel(result, selected_sheet, evaluation)
    st.download_button(
        label="Download results as Excel",
        data=excel_file,
        file_name="gl_account_recommendations.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        type="primary",
        width="stretch",
        on_click="ignore",
    )


if __name__ == "__main__":
    main()
