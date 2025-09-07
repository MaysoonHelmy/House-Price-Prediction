# House Price Prediction 

This project predicts housing prices using the **California Housing dataset**.
It compares **Linear Models (Linear Regression, Ridge, Lasso, ElasticNet)** with **Random Forest Regressors** (default and tuned).
The notebook demonstrates **data preprocessing, feature engineering, and model evaluation**.

---

## 📓 Notebook Workflow

1. **Data Preprocessing**

   * Dropped missing values.
   * Log-transformed skewed features (`total_rooms`, `total_bedrooms`, `population`, `households`).
   * Encoded `ocean_proximity` with **one-hot encoding** (`pd.get_dummies(..., dtype=int)`).
   * Scaled numeric features with **StandardScaler** for linear models.

2. **Feature Engineering**

   * Added informative ratio features:

     * `bedroom_ratio = total_bedrooms / total_rooms`
     * `household_rooms = total_rooms / households`
     * `population_per_household = population / households`

3. **Model Training & Evaluation**

   * **Linear Models**

     * Linear Regression baseline: **R² = 0.6687**
     * RidgeCV, LassoCV, ElasticNetCV all achieved \~**0.6688**
     * These models serve as interpretable baselines.

   * **Random Forest Regressor**

     * Default model: **R² = 0.8194**
     * Tuned with `RandomizedSearchCV`:

       ```python
       RandomForestRegressor(
           max_depth=20,
           min_samples_leaf=2,
           n_estimators=200,
           n_jobs=-1,
           random_state=42
       )
       ```

       Achieved **R² = 0.8217**

---

##  Results Summary

| Model                  | R² Score   |
| ---------------------- | ---------- |
| Linear Regression      | 0.6687     |
| RidgeCV                | 0.6688     |
| LassoCV                | 0.6688     |
| ElasticNetCV           | 0.6688     |
| RandomForest (default) | 0.8194     |
| RandomForest (tuned)   | **0.8217** |

Random Forest (both default and tuned) **clearly outperforms** all linear models.
 Linear models provide consistent **baselines** but cannot capture complex nonlinearities.

---

##  How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/house-price-prediction.git
   cd house-price-prediction
   ```

2. Open the Jupyter Notebook:

   ```bash
   jupyter notebook House_Price_Prediction.ipynb
   ```

---

##  Dependencies

* Python 3.8+
* pandas, numpy
* scikit-learn
* matplotlib, seaborn
* jupyter

---

## Project Highlights

* Compared **linear regression vs ensemble methods**.
* Applied **log transforms & ratio features** for better representation.
* Demonstrated **hyperparameter tuning** with `RandomizedSearchCV`.
* Showed that **Random Forest significantly outperforms Linear Regression variants**.

---

##  Acknowledgements

* Dataset: [California Housing Prices – Kaggle](https://www.kaggle.com/camnugent/california-housing-prices)
