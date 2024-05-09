import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

train_df = pd.read_csv('./datasets/train.csv')
test_df = pd.read_csv('./datasets/train.csv')

train_columns = set(train_df.columns)
test_columns = set(test_df.columns)

if train_columns != test_columns:
    missing_columns = train_columns - test_columns
    extra_columns = test_columns - train_columns
    print("Warning: Discrepancy in column names between training and test datasets.")
    print("Missing columns in test dataset:", missing_columns)
    print("Extra columns in test dataset:", extra_columns)
    test_df = test_df.drop(extra_columns, axis=1)
    for column in missing_columns:
        test_df[column] = pd.NA

numeric_features = train_df.select_dtypes(include=['int64', 'float64']).columns
categorical_features = train_df.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', IterativeImputer(max_iter=10, random_state=0)),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X = train_df.drop(['triglyceride_lvl', 'candidate_id'], axis=1)
y = train_df['triglyceride_lvl']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor())])

param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

print("Best parameters found: ", grid_search.best_params_)
best_model = grid_search.best_estimator_
val_predictions = best_model.predict(X_val)
val_mae = mean_absolute_error(y_val, val_predictions)
print('Validation Mean Absolute Error:', val_mae)

imputer = IterativeImputer(max_iter=10, random_state=0)
test_df_imputed = pd.DataFrame(imputer.fit_transform(test_df.drop('candidate_id', axis=1)), columns=test_df.columns[1:])

test_predictions = best_model.predict(test_df_imputed)

submission_df = pd.DataFrame({'candidate_id': test_df['candidate_id'], 'triglyceride_lvl': test_predictions})
submission_df.to_csv('./trained_data/filtered.csv', index=False)