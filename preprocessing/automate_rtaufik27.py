# Import library
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from imblearn.over_sampling import SMOTE

import logging
# Configure logging to write to a file and set the logging level
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# Create a logger
logger = logging.getLogger(__name__)


dataset = "namadataset_raw/employee_dataset_raw.csv"
df = pd.read_csv(dataset)

def run_preprocessing(df_pre):
    # Data Missing Handling
    df_pre['is_attrition_missing'] = df_pre['Attrition'].isna()

    # scaling
    scaler = MinMaxScaler()
    numeric_but_not = ['Attrition', 'EmployeeId', 'Education', 'EmployeeCount', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel', 'WorkLifeBalance']
    numeric_columns = df_pre.select_dtypes(include='number').drop(columns=numeric_but_not).columns
    df_pre[numeric_columns] = scaler.fit_transform(df_pre[numeric_columns])
    # Kolom numerik yang ingin diperlakukan sebagai kategorikal (tanpa target)
    non_numeric_columns = df_pre.select_dtypes(exclude='number').columns.union(
        pd.Index([col for col in numeric_but_not if col != 'Attrition'])
    )

    # Feature Selection
    df_feature = df_pre[df_pre['is_attrition_missing'] == False].copy()
    y = df_feature['Attrition']
    ## encoding categorical column ke numeric
    X = df_feature[numeric_columns.union(non_numeric_columns)].copy()
    for col in non_numeric_columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    ## Inisialisasi model dan CV
    estimator = RandomForestClassifier(random_state=42)
    cv = StratifiedKFold(5)
    ## RFECV otomatis cari jumlah fitur terbaik
    selector = RFECV(estimator, step=1, cv=cv, scoring='accuracy', n_jobs=-1)
    selector.fit(X, y)
    ## Ambil top 10 fitur
    selected_cols = X.columns[selector.support_]
    importances = selector.estimator_.feature_importances_
    feature_importance_list = list(zip(selected_cols, importances))
    sorted_features = sorted(feature_importance_list, key=lambda x: x[1], reverse=True)
    fix_selected_features = [feat[0] for feat in sorted_features[:10]]
    df_feature = df_feature[fix_selected_features + ['Attrition']]
    
    # Balancing
    ## Pisahkan fitur
    y = df_feature['Attrition']
    X = df_feature.drop(['Attrition'], axis=1)
    ## Encode semua fitur kategorikal dengan LabelEncoder
    X = X.copy()
    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col])
    ## Balancing data dengan SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Return Dataframe
    df_pre_final = pd.DataFrame(X_resampled, columns=X.columns)
    df_pre_final['Attrition'] = y_resampled

    return df_pre_final

run_preprocessing(df)
# print("===DONE===")
logger.info('===THE PROCESSING IS DONE===')