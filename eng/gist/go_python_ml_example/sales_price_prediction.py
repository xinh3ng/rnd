from sklearn.ensemble import RandomForestRegressor

# from sklearn.externals import joblib
import joblib


def predict_sale_price(MSSubClass, LotArea, YearBuilt, BedroomAbvGr, TotRmsAbvGrd):
    trained_model = joblib.load("model_backup(RF).joblib")
    prediction = trained_model.predict([[MSSubClass, LotArea, YearBuilt, BedroomAbvGr, TotRmsAbvGrd]])
    return prediction
