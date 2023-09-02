import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error as mae

data = pd.read_csv('combined_features.csv')
audio_data = pd.read_csv('audio_features.csv')
text_data = pd.read_csv('text_features.csv')

mse_lm_list = []
mse_dummy_list = []
mse_gb_list = []
mse_rf_list = []

r2_lm_list = []
r2_dummy_list = []
r2_gb_list = []
r2_rf_list = []
r_state_list = []

mae_lm_list = []
mae_dummy_list = []
mae_gb_list = []
mae_rf_list = []

random_state=0

results_dummy = np.zeros((30))
results_linear = np.zeros((30))
results_gb = np.zeros((30))
results_rf = np.zeros((30))

for ramdom_state in range(0, 30):
    r_state_list.append(ramdom_state)
    """ X = data.drop(columns=['tot_scale', 'class','ID'])
    y = data['tot_scale']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state) """
    X = data.drop(columns=['tot_scale','class','ID'])
    y = data['tot_scale']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=868, shuffle=False, random_state=random_state)
    """ X_text = text_data.drop(columns=['tot_scale'])
    y_text = text_data['tot_scale']
    X_t_train, X_t_test, y_t_train, y_t_test = train_test_split(X_test, y_text, test_size=0.2, random_state=random_state) """


    lm = LinearRegression().fit(X_train, y_train)
    lm_dummy_mean = DummyRegressor(strategy = 'mean').fit(X_train, y_train)
    y_predict = lm.predict(X_test)
    y_predict_dummy_mean = lm_dummy_mean.predict(X_test)
    mae_dummy = mae(y_test, y_predict_dummy_mean)
    results_dummy[random_state] = mae_dummy
    mae_dummy_list.append(mae_dummy)
    mse_dummy_list.append(mean_squared_error(y_test, y_predict_dummy_mean))
    r2_dummy_list.append(r2_score(y_test, y_predict_dummy_mean))
    
    mse = mean_squared_error(y_test, y_predict)
    r2 = r2_score(y_test, y_predict)
    mse_lm_list.append(mse)
    r2_lm_list.append(r2)
    mae_lm = mae(y_test,y_predict)
    results_linear[random_state] = mae_lm
    mae_lm_list.append(mae_lm)


    gb_model = GradientBoostingRegressor(n_estimators=100)
    gb_model.fit(X_train, y_train)
    # Make predictions
    y_pred = gb_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mse_gb_list.append(mse)
    r2_gb_list.append(r2)
    mae_gb = mae(y_test, y_pred)
    results_gb[ramdom_state] = mae_gb
    mae_gb_list.append(mae_gb)

    # Create and fit the Random Forest classifier
    rf_model = RandomForestRegressor(n_estimators=100)
    rf_model.fit(X_train, y_train)
    # Make predictions
    y_pred = rf_model.predict(X_test)
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mse_rf_list.append(mse)
    r2_rf_list.append(r2)
    mae_rf = mae(y_test, y_pred)
    results_rf[ramdom_state] = mae_rf
    mae_rf_list.append(mae_rf)
    
    
    
t_statistic, p_value = stats.ttest_rel(results_dummy, results_linear)
print("Linear: t ", t_statistic)
print("Linear: p ", p_value)

t_statistic, p_value = stats.ttest_rel(results_dummy, results_gb)
print("Gradient Boosting: t ", t_statistic)
print("Gradient Boosting: p ", p_value)

t_statistic, p_value = stats.ttest_rel(results_dummy, results_rf)
print("Random Forest: t ", t_statistic)
print("Random Forest: p ", p_value)

df = pd.DataFrame(np.column_stack([r_state_list, mse_dummy_list, r2_dummy_list, mae_dummy_list,mse_lm_list ,r2_lm_list, mae_lm_list,  mse_gb_list, r2_gb_list, mae_gb_list,mse_rf_list, r2_rf_list, mae_rf_list]),
                               columns=['random_state', 'dummy_mse', 'dummy_r2', 'dummy_mae','linear_mse', 'linear_r2',  'linear_mae', 'mse_gb', 'r2_gb',  'mae_gb','mse_rf', 'r2_rf',  'mae_rf' ])  #add these lists to pandas in the right order
# Write out the updated dataframe
df.to_csv("combined_results_not_shuffle.csv", index=False)