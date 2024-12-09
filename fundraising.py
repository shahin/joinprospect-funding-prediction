import pandas as pd
import ast
from datetime import datetime
import tempfile
from typing import Dict, Tuple, Any

import modal
import modal.gpu
import treelite


dtype_dict = {
    'deal_id': str,
    'pb_id': str,
    'name': str,
    'deal_number': int,
    'year_founded': 'Int64',
    'deal_size': float,
    'post_valuation': float,
    'total_vc_funding': float,
    'total_invested_equity': float,
    'deal_type': str,
    'deal_type2': str,
    'vc_round': str,
    'lead_investor': str,
    'company_website': str,
    'hq_location': str,
    'description': str,
    'primary_industry_code': str,
    'primary_industry_group': str,
    'primary_industry_sector': str
}


def parse_list(x):
    if pd.isna(x) or x == '[]':
        return []
    try:
        return ast.literal_eval(x)
    except:
        return []


def parse_date(x):
    if pd.isna(x):
        return None
    try:
        return datetime.strptime(x, '%Y-%m-%d')
    except:
        return None


def read_data(path: str):

    df = pd.read_csv(path,  
                     sep='\t',
                     dtype=dtype_dict,  # type: ignore
                     converters={
                         'deal_date': parse_date,
                         'investors': parse_list,
                         'investor_ids': parse_list,
                         'first_time_investors': parse_list,
                         'keywords': parse_list,
                         'verticals': parse_list
                     },
                     na_values=['', 'nan', 'NULL'])
    
    df['post_valuation_status'] = df['post_valuation_status'].astype('category')
    return df


def convert_vc_round(round_str):
    if pd.isna(round_str) or not round_str:
        return None
    
    mapping = {
        '1st': 1,
        '2nd': 2,
        '3rd': 3,
        '4th': 4,
        '5th': 5,
        '6th': 6,
        '7th': 7,
        '8th': 8,
        '9th': 9,
        '10th': 10,
        'Angel': 0,
    }
    
    round_lower = round_str.lower()
    for key in mapping:
        if key in round_lower:
            return mapping[key]
    return None


def get_early_stage_deals(df: pd.DataFrame, threshold_millions: int) -> pd.DataFrame:
    # Sort by date and calculate cumulative funding
    df_sorted = df.sort_values(['pb_id', 'deal_number'])
    df_sorted['cumulative_funding'] = df_sorted.groupby('pb_id')['deal_size'].cumsum()
    
    # Get first deal that takes the firm over the threshold
    result = (df_sorted[df_sorted['cumulative_funding'] >= threshold_millions]
              .groupby('pb_id')
              .first()
              .reset_index())
    
    return result


def has_complete_deals(deals):
        expected_range = range(1, max(deals) + 1)
        return set(deals) == set(expected_range)


def remove_incomplete_firms(df: pd.DataFrame) -> pd.DataFrame:
    complete_firms = df.groupby('pb_id')['deal_number'].agg(has_complete_deals)
    return df[df['pb_id'].isin(complete_firms[complete_firms].index)]


def add_post_accelerator(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Find accelerator deals for each firm
    accelerator_dates = df[
        (df['deal_type'] == 'Accelerator/Incubator')
    ].groupby('pb_id')['deal_date'].min()

    # Mark deals that came after accelerator
    df['post_accelerator'] = False
    for pb_id, acc_date in accelerator_dates.items():
        df.loc[
            (df['pb_id'] == pb_id) &
            (df['deal_date'] > acc_date),
            'post_accelerator'
        ] = True

    return df


rapids_image = modal.Image.from_registry("rapidsai/base:24.12a-cuda12.0-py3.12")
app = modal.App('joinprospect')

@app.function(gpu=modal.gpu.A100(size="80GB"), image=rapids_image)
def predict_deal_size(df: pd.DataFrame, n_folds = 5) -> tuple:

    # import RAPIDS packages (cuml, cudf) inside function scope because they're required
    # and available on remote Modal images but neither required nor available locally on Mac
    from cuml.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import MultiLabelBinarizer
    import numpy as np
    import cudf
    import cupy as cp
    from typing import Dict, Tuple, Any, List
    from scipy import stats

    df = cudf.from_pandas(df)

    def encode_categorical(series: cudf.Series) -> cp.ndarray:
        """Encode categorical variable using cuDF's categorical dtype"""
        cat_series = series.astype('category')
        # get the codes as a numpy array first to avoid cuDF masked array errors
        codes = cat_series.cat.codes.to_numpy()
        return cp.array(codes, dtype=np.float32).reshape(-1, 1)

    def convert_multilabel_to_gpu(labels: List[List[str]], mlb: MultiLabelBinarizer) -> cp.ndarray:
        """Convert multi-label features to GPU-compatible format"""
        # Fit and transform on CPU
        cpu_matrix = mlb.fit_transform(labels)
        # Transfer to GPU
        return cp.array(cpu_matrix)

    # Initialize CPU-side multi-label binarizers
    mlb_keywords = MultiLabelBinarizer()
    mlb_verticals = MultiLabelBinarizer()
    
    # Transform categorical features using cuDF's native categorical type
    X_location = encode_categorical(df['hq_location'])
    X_industry = encode_categorical(df['primary_industry_code'].fillna('UNKNOWN'))
    
    # Convert lists to CPU for multi-label processing
    keywords_list = df['keywords'].to_pandas().tolist()
    verticals_list = df['verticals'].to_pandas().tolist()
    
    X_keywords = convert_multilabel_to_gpu(keywords_list, mlb_keywords)
    X_verticals = convert_multilabel_to_gpu(verticals_list, mlb_verticals)
    
    # Convert numerical features to GPU arrays
    X_numerical = cp.array(df[['post_accelerator', 'cumulative_funding']].to_numpy(), dtype=np.float32)
    
    X = cp.hstack([
        X_location,
        X_industry,
        X_keywords,
        X_verticals,
        X_numerical
    ])
    
    y = cp.array(df['deal_size'].to_numpy(), dtype=np.float32)
    
    rf_params = {
        "n_estimators": 10,
        "max_depth": 128,
        "n_bins": 256,
    }
    print(f"Random forest regressor params: {rf_params}")
    rf = RandomForestRegressor(
        random_state=42,
        n_streams=1,
        **rf_params,
    )
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    X_np = cp.asnumpy(X)  # convert to numpy for sklearn's KFold

    cv_scores = {'r2': [], 'rmse': [], 'mae': []}
    
    for train_idx, val_idx in kf.split(X_np):
        # Convert indices back to GPU arrays
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train and predict
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        
        # Calculate metrics on CPU
        y_val_np = cp.asnumpy(y_val)
        y_pred_np = cp.asnumpy(y_pred)
        
        cv_scores['r2'].append(np.corrcoef(y_val_np, y_pred_np)[0, 1] ** 2)
        cv_scores['rmse'].append(np.sqrt(np.mean((y_val_np - y_pred_np) ** 2)))
        cv_scores['mae'].append(np.mean(np.abs(y_val_np - y_pred_np)))

    # Train the model on all the data
    rf.fit(X, y)
    
    # Make predictions
    y_pred = rf.predict(X)
    residuals = cp.asnumpy(y - y_pred)

    # Check distribution of residuals
    normality_test = stats.normaltest(residuals)
    
    # Get feature names (using categorical codes now)
    feature_names = (
        [f'location_{i}' for i in range(X_location.shape[1])] +
        [f'industry_{i}' for i in range(X_industry.shape[1])] +
        mlb_keywords.classes_.tolist() +
        mlb_verticals.classes_.tolist() +
        ['post_accelerator', 'cumulative_funding']
    )

    checkpoint_path = "./checkpoint.tl"
    print("Writing random forest to treelite model checkpoint ...")
    treelite_model = rf.convert_to_treelite_model().to_treelite_checkpoint(checkpoint_path)
    with open(checkpoint_path, "rb") as f:
        treelite_model_serialized = f.read()
    print(f"Read {len(treelite_model_serialized)} bytes")
    
    results = {
        'model': treelite_model_serialized,
        'cross_validation': {
            'r2_scores': np.array(cv_scores['r2']),
            'r2_mean': np.mean(cv_scores['r2']),
            'r2_std': np.std(cv_scores['r2']),
            'rmse_scores': np.array(cv_scores['rmse']),
            'rmse_mean': np.mean(cv_scores['rmse']),
            'rmse_std': np.std(cv_scores['rmse']),
            'mae_scores': np.array(cv_scores['mae']),
            'mae_mean': np.mean(cv_scores['mae']),
            'mae_std': np.std(cv_scores['mae'])
        },
        'residual_analysis': {
            'residuals': residuals,
            'normality_test_statistic': normality_test.statistic,
            'normality_test_p_value': normality_test.pvalue
        },
        # Note: we're not storing encoders anymore since we're using cuDF's categorical type
        'category_maps': {
            'location': dict(enumerate(df['hq_location'].astype('category').cat.categories.values_host)),
            'industry': dict(enumerate(df['primary_industry_code'].astype('category').cat.categories.values_host)),
            'keywords': dict(enumerate(mlb_keywords.classes_)),
            'verticals': dict(enumerate(mlb_verticals.classes_))
        }
    }
    
    return results


@app.local_entrypoint()
def main():

    tsv_path = "./fundraising.tsv"
    print(f"Reading {tsv_path} ...")
    df = read_data(path=tsv_path)
    print(f"... read {len(df)} rows")

    # remove Accellerator deals (optionally add an indicator var to the firm)
    accelerator_ind_df = add_post_accelerator(df)
    print(f"len(accelerator_ind_df)={len(accelerator_ind_df)}")

    # remove firms with missing deals
    complete_firms_df = remove_incomplete_firms(accelerator_ind_df)
    print(f"len(complete_firms_df)={len(complete_firms_df)}")

    non_accellerator_deals_df = accelerator_ind_df[~accelerator_ind_df['deal_id'].isin(
        accelerator_ind_df[accelerator_ind_df.deal_type == 'Accelerator/Incubator']['deal_id']
    )]
    print(f"len(non_accellerator_deals_df)={len(non_accellerator_deals_df)}")

    # remove firms with deals that have missing values
    firms_with_known_deal_sizes_df = non_accellerator_deals_df[non_accellerator_deals_df['deal_size'].notna()]
    print(f"len(firms_with_known_deal_sizes_df)={len(firms_with_known_deal_sizes_df)}")

    # subset to early stage deals
    early_stage_deals = get_early_stage_deals(firms_with_known_deal_sizes_df, 10)
    print(f"len(early_stage_deals)={len(early_stage_deals)}")

    results = predict_deal_size.remote(early_stage_deals)

    treelite_model_file = tempfile.NamedTemporaryFile(delete=False, delete_on_close=False)
    with open(treelite_model_file.name, "wb") as treelite_file:
        treelite_file.write(results['model'])
    print(f"Wrote {len(results['model'])} bytes to {treelite_model_file.name}")
    results['model'] = treelite.Model.deserialize(treelite_model_file.name)
    print(results)


if __name__ == "__main__":
    main()
