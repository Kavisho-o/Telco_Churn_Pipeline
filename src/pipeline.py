from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier


def build_pipeline(continuous_cols, categorical_cols, binary_numeric_cols):

    preprocessor = ColumnTransformer(
        transformers=[
            ("cont", StandardScaler(), continuous_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
            ("bin", "passthrough", binary_numeric_cols)
        ]
    )

    model = GradientBoostingClassifier(
        subsample=0.6,
        n_estimators=400,
        min_samples_split=2,
        min_samples_leaf=4,
        max_depth=4,
        learning_rate=0.01,
        random_state=42
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline