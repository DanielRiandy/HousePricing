import pandas as pd 
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

class algorithm:
    def __init__(self, algo_) -> None:
        self.model= algo_


    def evaluate_(self, X: pd.DataFrame, y:pd.Series , 
                  scaling: bool = True, SaveArtifacts:bool = False):
        ## Splitting
        X_train, X_test, y_train, y_test = train_test_split(
                                            X, y, test_size=.2,
                                            random_state=42)
        
        ## Vectorizer
        dv = DictVectorizer()

        train_dicts = X_train.to_dict(orient = 'records')
        X_train_vect = dv.fit_transform(train_dicts)

        test_dicts = X_test.to_dict(orient = 'records')
        X_test_vect = dv.transform(test_dicts)

        ## Scalling
        if scaling:
            scaler_ = StandardScaler()
            X_train = scaler_.fit_transform(X_train)
            X_test = scaler_.transform(X_test)

        ## Training phase
        model = self.model
        model.fit(X_train_vect, y_train)

        ## Evaluate phase
        train_pred = model.predict(X_train_vect)
        test_pred = model.predict(X_test_vect)

        train_mae, train_r2 = mean_absolute_error(y_train, train_pred), r2_score(y_train, train_pred)
        test_mae, test_r2 = mean_absolute_error(y_test, test_pred), r2_score(y_test, test_pred)

        print(f'''
Result:
    MAE Train = {train_mae}
    MAE Test = {test_mae}

    R2 Train = {train_r2}
    R2 Test = {test_r2}
        ''')

        if SaveArtifacts:
            os.makedirs("./Artifacts/", exist_ok=True)
            with open('./Artifacts/model.bin', 'wb') as f_out:
                pickle.dump(model, f_out)
            with open('./Artifacts/features.bin', 'wb') as f_out:
                pickle.dump(X_test.columns.tolist(), f_out)
            with open('./Artifacts/vectorizer.bin', 'wb') as f_out:
                pickle.dump(dv, f_out)

        ## Result Viz
        pca = PCA(n_components=1, random_state=42)
        train_ = pca.fit(X_train)
        test_ = pca.transform(X_test)
        temp = pd.concat([pd.DataFrame(test_).reset_index(drop = True), 
                pd.DataFrame(y_test).reset_index(drop = True),
                pd.DataFrame(test_pred).reset_index(drop = True)], axis = 1)
        temp.columns = ["X", "y", "y_pred"]
        temp = temp.sort_values('X')


        plt.plot(temp['X'], temp['y'], 'ro', label = 'Actual Price' )
        plt.plot(temp['X'], temp['y_pred'], 'b', label = 'Prediction')
        plt.title("Prediction vs Actual")
        plt.ylabel("Price")
        plt.xlabel("Features")
        plt.legend()
        plt.show()

        return model



