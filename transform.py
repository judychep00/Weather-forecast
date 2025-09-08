import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.impute import SimpleImputer

class ZindiCleaner:
    def __init__(self):
        self.pipeline = None
        self.categorical_features = ['community', 'district', 'indicator']
        self.numerical_features = ['confidence', 'predicted_intensity', 'forecast_length', 'year', 'month', 'day']

    def _extract_date_features(self, df):
        if 'prediction_time' not in df.columns:
            raise ValueError("Missing 'prediction_time' column in input DataFrame.")
        df['prediction_time'] = pd.to_datetime(df['prediction_time'], errors='coerce')
        df['year'] = df['prediction_time'].dt.year
        df['month'] = df['prediction_time'].dt.month
        df['day'] = df['prediction_time'].dt.day
        return df

    def _create_preprocessor(self):
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        preprocessor = ColumnTransformer(transformers=[
            ('num', numerical_transformer, self.numerical_features),
            ('cat', categorical_transformer, self.categorical_features)
        ])
        return preprocessor

    def build_pipeline(self):
        self.pipeline = Pipeline(steps=[
            ('preprocessor', self._create_preprocessor())
        ])

    def transform(self, df):
        df = self._extract_date_features(df)
        X = df.drop(columns=['ID', 'indicator_description', 'time_observed'], errors='ignore')
        self.build_pipeline()
        return self.pipeline.fit_transform(X)
        

    
    
    