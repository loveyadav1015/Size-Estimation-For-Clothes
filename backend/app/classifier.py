import joblib
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SIZE_ESTIMATOR_MODEL_PATH = BASE_DIR.parent / "models" / "clothing_model.pkl"

try:
    rf_model = joblib.load(str(SIZE_ESTIMATOR_MODEL_PATH))
    print("Clothing Model Loaded Successfully")
except:
    print("ERROR: Run the training script first!")
# when generating the data instead of using s,m,l and xl i told gemini to use 0,1,2 and 3. now this is because models prefer to deal with numbers and many relate these numbers also. like 1 is more close 2 then 3, which is true as med is more close to large than to xl. this also helps when calculating accuracy   
SIZE_MAP = {0: 'S', 1: 'M', 2: 'L', 3: 'XL'}

def get_estimated_size(df_features):
    prediction_code = rf_model.predict(df_features)[0]
            
    predicted_size = SIZE_MAP[prediction_code]

    return predicted_size