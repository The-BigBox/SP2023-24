import os
import re
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import ARIMA
import matplotlib.pyplot as plt
from darts.metrics import mape, mse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from darts.dataprocessing.transformers import Scaler
from darts.models import TransformerModel, BlockRNNModel, NBEATSModel, XGBModel, LightGBMModel

ORIGINAL_DATA_PATH = os.getcwd() + '/data/Fundamental+Technical Data/STOCK_DATA/'
DATA_PATH = os.getcwd() + '/data/Fundamental+Technical Data/STOCK_DATA_WEEKLY/'
LDA_NEWS_PATH = os.getcwd() + '/data/Online Data/LDA News/'
ID_PATH = os.getcwd() + '/data/Fundamental+Technical Data/ID_Name.csv'
RESULT_PATH = os.getcwd() + '/result/'
PARAMETER_PATH = os.getcwd() + '/model/'
FOCUS_COMPONENT = 'Close'
RETAIN_COMPONENTS = ["Open", "High", "Low", "PE", "PBV", "T_EPS", "FSCORE", "Vol",
                           "Buy Vol", "Sell Vol", "ATO/ATC", "EMA25", "EMA50", "EMA200", "MACD", "RSI"]
TRAINING_SIZE = 139
VALIDATE_SIZE = 35
TEST_SIZE = 35
PREDICT_SIZE = 1
STOCK_LIST = ["ADVANC", "BH", "CBG", "CPALL", "INTUCH", "IVL", "PTT", "PTTEP", "PTTGC", "SCB", "SCC", "TISCO", "WHA"]
MODEL_PARAM = {
    # 'TransformerModel': {
    #     'input_chunk_length': [1,3,5], 
    #     'output_chunk_length': [1], 
    #     'n_epochs': [15],
    #     # Add more Transformer-specific parameters if needed
    # },
    # 'BlockRNNModel': {
    #     'model': ['LSTM'],
    #     'input_chunk_length': [1,3,5], 
    #     'output_chunk_length': [1], 
    #     'n_epochs': [15],
    #     # Add more BlockRNN-specific parameters if needed
    # },
    # 'NBEATSModel': {
    #     'input_chunk_length': [1,3,5], 
    #     'output_chunk_length': [1], 
    #     'n_epochs': [15],
    #     # Add more N-BEATS-specific parameters if needed
    # }

    'XGBModel': {
        'lags': [1,3,5], 
        'lags_past_covariates': [1,3,5], 
        'output_chunk_length': [1], 
    }
}

def convert_weekly_data(stock_name):
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH) 
    
    # Read the data, initially parsing dates as objects
    stock_data = pd.read_csv(ORIGINAL_DATA_PATH + stock_name + ".csv")
    
    # Convert the 'Date' column to datetime using the specified format
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    
    # Select every 7th data point from stock data
    stock_data = stock_data.iloc[::7, :]
    
    stock_data.to_csv(DATA_PATH + stock_name + ".csv", index=False)
    
def highest_numbered_folder(directory):
    number_regex = re.compile(r'\b\d+\b')

    max_number = -1
    max_folder_path = None

    for item in os.listdir(directory):
        full_path = os.path.join(directory, item)
        if os.path.isdir(full_path):
            match = number_regex.search(item)
            if match:
                number = int(match.group())
                if number > max_number:
                    max_number = number
                    max_folder_path = full_path

    return max_folder_path
    
def load_data(stock_name):
    # Read the data, initially parsing dates as objects
    stock_data = pd.read_csv(os.path.join(DATA_PATH, (stock_name + ".csv")))
    id_name_map = pd.read_csv(ID_PATH)
    stock_id = id_name_map.loc[id_name_map['Stock_Name'].str.strip() == stock_name, 'Stock_ID'].iloc[0]
    industry_name = id_name_map.loc[id_name_map['Stock_Name'].str.strip() == stock_name, 'Industry_Name'].iloc[0]
    highest_folder_path = highest_numbered_folder(LDA_NEWS_PATH)
    lda_news_df = "" #pd.read_csv(f"{highest_folder_path}/{industry_name}.csv", dayfirst=True, parse_dates=["Date"])
    lda_twitter_df = ""
    GDELTv1 = ""
    GDELTv2 = ""
    data_df = stock_data.copy()
    return stock_data, stock_id, data_df, lda_news_df, lda_twitter_df, GDELTv1, GDELTv2

def preprocess_lda_news(data_df, lda_news_df):
    # Assuming you've already read the datasets as provided
    data_df.set_index('Date', inplace=True)
    lda_news_df.set_index('Date', inplace=True)

    # Identify dates in lda_news_df that are not in data_df
    non_intersecting_dates = lda_news_df.index.difference(data_df.index)

    # Iterate through the non-intersecting dates
    for date in non_intersecting_dates:
        # Find the closest previous date in lda_news_df that also exists in data_df
        previous_dates = lda_news_df.index[(lda_news_df.index < date) & (lda_news_df.index.isin(data_df.index))]
        
        if not previous_dates.empty:
            nearest_prev_date = previous_dates[-1]
            
            # Accumulate values from the non-matching date to the closest previous date
            for column in lda_news_df.columns:
                lda_news_df.at[nearest_prev_date, column] += lda_news_df.at[date, column]
            
            # Drop the non-matching date from lda_news_df
            lda_news_df.drop(date, inplace=True)
    # Filter lda_news_df to only include dates present in data_df
    updated_lda_news_df = lda_news_df[lda_news_df.index.isin(data_df.index)]
    return updated_lda_news_df

def preprocess_data(data, data_df,  lda_news_df, lda_twitter_df, GDELTv1, GDELTv2, split, features):
    data = data.dropna()
    serie = data[FOCUS_COMPONENT]
    past_covariate = data[RETAIN_COMPONENTS].apply(pd.to_numeric, errors='coerce').ffill().bfill()
    
    if 2 in features:
        updated_lda_news_df = preprocess_lda_news(data_df, lda_news_df)
        past_covariate = past_covariate.join(updated_lda_news_df.reset_index().drop(columns='Date'))

    serie_ts = TimeSeries.from_dataframe(serie.to_frame())
    past_cov_ts = TimeSeries.from_dataframe(past_covariate)
    scaler = StandardScaler()
    scaler_dataset = Scaler(scaler)
    scaled_serie_ts = scaler_dataset.fit_transform(serie_ts)
    if split == 0:
        training_scaled = scaled_serie_ts
    else:
        training_scaled = scaled_serie_ts[:-split]
        past_cov_ts = past_cov_ts[:-split]
    return training_scaled, past_cov_ts, scaler_dataset

def predict_next_n_days(model, training_scaled, past_cov_ts, scaler_dataset):
    """Predict next n days' closing prices for each stock."""
    model.fit(training_scaled, past_covariates=past_cov_ts, verbose=True)
    forecast = model.predict(PREDICT_SIZE, verbose=True)
    in_forecast = scaler_dataset.inverse_transform(forecast)
    return in_forecast

def generate_output(filename, predictions, stock_data, stock_id, split, stock, filepath):
    # Generate output
    # Handle the date mapping for predictions
    if split - PREDICT_SIZE <= 0: 
        last_known_date = pd.to_datetime(stock_data['Date'].iloc[-1], dayfirst=True)
        
        if split == 0:  # Add this condition
            difference = PREDICT_SIZE
            future_dates = [(last_known_date + pd.Timedelta(days=i+1)).strftime('%-d/%-m/%Y') for i in range(difference)]
            date = future_dates
        else:
            difference = -1 * (split - PREDICT_SIZE)
            future_dates = [(last_known_date + pd.Timedelta(days=i+1)).strftime('%-d/%-m/%Y') for i in range(difference)]
            date = stock_data['Date'][-split:].tolist() + future_dates
    else:
        date = stock_data['Date'][- split:len(stock_data) - split+PREDICT_SIZE].reset_index(drop=True)

    prediction_df = predictions.pd_dataframe().reset_index(drop=True)
    date_df = pd.DataFrame(date, columns=['Date'])
    combined_df = pd.concat([date_df, prediction_df], axis=1).reset_index(drop=True)
    
    # Fill missing values in predictions
    combined_df.fillna(0, inplace=True)
    
    # Format the predictions into the desired output.
    output = []
    predict_date = stock_data['Date'].iloc[-split-1]
    # Get stock_id using the dictionary; if not found, raise an error
    
    for _, row in combined_df.iterrows():
        date = row.iloc[0]
        pred_value = round(row.iloc[1], 2)
        output.append([predict_date, stock_id, date, pred_value])
            
    output_df = pd.DataFrame(output, columns=["Predict_Date", "Stock_ID", "Date", "Closing_Price"])
    output_df.to_csv(filepath+filename+".csv", mode='a', header=not os.path.exists(filepath+filename+".csv"))

def finalize_csv(csv_path):
    final_df = pd.read_csv(csv_path)
    
    # Drop old index column if it exists
    if 'Unnamed: 0' in final_df.columns:
        final_df.drop('Unnamed: 0', axis=1, inplace=True)
    
    final_df.reset_index(inplace=True)
    final_df.rename(columns={'index': 'Order_ID'}, inplace=True)
    final_df.to_csv(csv_path, index=False)

def arima_prediction(train, val):
    # Create and fit an ARIMA model
    model = ARIMA(p = 1, d = 1, q = 1)
    model.fit(train)

    # Forecast the next steps (same size as the validation set, which is 60)
    predicted = model.predict(len(val))
    return predicted

def moving_average(stock):
    if not os.path.exists(f'{PARAMETER_PATH}/{stock}/Moving Average'):
        os.makedirs(f'{PARAMETER_PATH}/{stock}/Moving Average')

    window_size = [5, 10, 15, 20]

    for size in window_size:
        stock_data, stock_id, _, _, _, _, _ = load_data(stock)
        df = stock_data[['Date', 'Close']].iloc[TRAINING_SIZE-size:-TEST_SIZE]
        df['Moving_Average'] = df['Close'].rolling(window=size).mean()
        df = df.dropna()
        output = pd.DataFrame()
        output['Predict_Date'] = df['Date'].reset_index(drop=True)
        output['Stock_ID'] = stock_id
        output['Date'] = df['Date'].iloc[1:].reset_index(drop=True)
        output['Closing_Price'] = df['Moving_Average'].round(2).reset_index(drop=True)
        output.index.names = ['Prediction']
        output = output.dropna()

        output.to_csv(f'{PARAMETER_PATH}/{stock}/Moving Average/window_size_{size}.csv')
    print("Finished caculate Moving Average for ", stock)

def directional_accuracy(actual, forecasted):
    # Extract values from TimeSeries objects for computation
    actual = actual.values().flatten()
    forecasted = forecasted.values().flatten()
    
    # Calculate day-to-day differences
    actual_diff = actual[1:] - actual[:-1]
    forecasted_diff = forecasted[1:] - forecasted[:-1]

    # Calculate the sign of the differences
    actual_sign = np.sign(actual_diff)
    forecasted_sign = np.sign(forecasted_diff)

    # Calculate number of days where the direction matches
    matches = (actual_sign == forecasted_sign).sum()

    return matches / len(actual_diff) * 100

def plot_graph(dates, series, val, arima_pred, new_model_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(dates[-300:-60], series.values()[-300:-60], label='Actual (Last 300 days)')
    plt.plot(dates[-60:], val.values(), label='Actual (Last 60 days)', color='blue')
    
    last_actual_value = series.values()[-61]
    # Append the last actual value to the beginning of the prediction series
    arima_pred_with_last = np.insert(arima_pred.values(), 0, last_actual_value)
    new_model_pred_with_last = np.insert(new_model_pred.values(), 0, last_actual_value)

    # Plot the forecasted values (including the last actual value for a smooth transition)
    plt.plot(dates[-61:], arima_pred_with_last, label='ARIMA Forecast', lw=2, color='red')
    plt.plot(dates[-61:], new_model_pred_with_last, label='New Model Forecast', lw=2, color='green')


    plt.title('Stock Price Prediction using ARIMA vs New Model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='upper left', bbox_to_anchor=(1,1))
    plt.show()

def cal_err_and_acc(val, predicted, condition=True):
    val = val.pd_dataframe()
    val.reset_index(drop=True, inplace=True)
    predicted = predicted.pd_dataframe()
    predicted.reset_index(drop=True, inplace=True)
    val_ts = TimeSeries.from_dataframe(val)
    predicted_ts = TimeSeries.from_dataframe(predicted)

    mape_error = mape(val_ts, predicted_ts)
    rmse_error = np.sqrt(mse(val_ts, predicted_ts))
    dir_acc = directional_accuracy(val_ts, predicted_ts)
    if condition:
        return mape_error, rmse_error, dir_acc
    else:
        print(f"Directional Accuracy = {dir_acc:.2f} %")
        print(f"MAPE = {mape_error:.2f} %")
        print(f"RMSE = {rmse_error:.2f} %\n")

def find_best_param(stock_name):
    path = PARAMETER_PATH + stock_name
    all_dir = os.listdir(path)
    for folder_list in all_dir:
        if folder_list == ".DS_Store":
            continue
        best_params_by_dir = {'param': None, 'da': 0, 'mape': float('inf'), 'rmse': float('inf')}
        best_params_by_mape = {'param': None, 'da': 0, 'mape': float('inf'), 'rmse': float('inf')}
        best_params_by_rmse = {'param': None, 'da': 0, 'mape': float('inf'), 'rmse': float('inf')}
        check_path = path + "/" + folder_list
        dir_list = os.listdir(check_path)
        for param_file in dir_list:
            # Skip non-CSV files
            if not param_file.endswith('.csv'):
                continue
            # Read the file
            predict = pd.read_csv(check_path + '/' + param_file)
            predict_ts = TimeSeries.from_dataframe(predict[['Closing_Price']], time_col=None)
            val = pd.read_csv(DATA_PATH + stock_name + ".csv")
            val = val[['Close']].iloc[-MAX_SPLIT_SIZE:]
            val_ts = TimeSeries.from_dataframe(val, time_col=None)    

            avg_mape, avg_rmse, avg_dir = cal_err_and_acc(predict_ts, val_ts, True)

            # Update best parameters
            if avg_dir > best_params_by_dir['da']:
                best_params_by_dir.update({'param': param_file, 'da': avg_dir, 'mape': avg_mape, 'rmse': avg_rmse})
            if avg_mape < best_params_by_mape['mape']:
                best_params_by_mape.update({'param': param_file, 'da': avg_dir, 'mape': avg_mape, 'rmse': avg_rmse})
            if avg_rmse < best_params_by_rmse['rmse']:
                best_params_by_rmse.update({'param': param_file, 'da': avg_dir, 'mape': avg_mape, 'rmse': avg_rmse})

        # Save the best parameters to a text file
        with open(check_path + '/best_parameters.txt', 'w') as f:
            f.write("Best Dir Param:\n")
            f.write(f"Param: {best_params_by_dir['param']}\n")
            f.write(f"Dir: {round(best_params_by_dir['da'], 2)}\n")
            f.write(f"MAPE: {round(best_params_by_dir['mape'], 2)}\n")
            f.write(f"RMSE: {round(best_params_by_dir['rmse'], 2)}\n\n")

            f.write("Best MAPE Param:\n")
            f.write(f"Param: {best_params_by_mape['param']}\n")
            f.write(f"Dir: {round(best_params_by_mape['da'], 2)}\n")
            f.write(f"MAPE: {round(best_params_by_mape['mape'], 2)}\n")
            f.write(f"RMSE: {round(best_params_by_mape['rmse'], 2)}\n\n")

            f.write("Best RMSE Param:\n")
            f.write(f"Param: {best_params_by_rmse['param']}\n")
            f.write(f"Dir: {round(best_params_by_rmse['da'], 2)}\n")
            f.write(f"MAPE: {round(best_params_by_rmse['mape'], 2)}\n")
            f.write(f"RMSE: {round(best_params_by_rmse['rmse'], 2)}\n")        
    
def stock_tuning(stock_name, features):
    # Check if the daily tuning results file exists and remove it if it does
    if not os.path.exists(PARAMETER_PATH + stock_name):
        os.makedirs(PARAMETER_PATH + stock_name) 
    
    print(f"Tuning {stock_name} ...")
 
    for model_type, params_grid in MODEL_PARAM.items():
        for params in ParameterGrid(params_grid):
            if model_type == 'TransformerModel':
                model = TransformerModel(**params)
            elif model_type == 'BlockRNNModel':
                model = BlockRNNModel(**params)
            elif model_type == 'NBEATSModel':
                model = NBEATSModel(**params)
            elif model_type == 'XGBModel':
                model = XGBModel(**params)

            generate_path = PARAMETER_PATH+f"{stock_name}/"
            if 1 in features:
                generate_path = generate_path+"/Fundamental"
            if 2 in features:
                generate_path = generate_path+"+LDA News"
            if 3 in features:
                generate_path = generate_path+"+LDA Twitter"
            if 4 in features:
                generate_path = generate_path+"+GDELT V1"
            if 5 in features:
                generate_path = generate_path+"+GDELT V2"
                
            if not os.path.exists(generate_path):
                os.makedirs(generate_path) 

            # Generate a filename from parameters
            params_str = '_'.join(f"{key}{val}" for key, val in params.items())
            filename = f"{model_type}_{params_str}"

            if os.path.exists(generate_path+"/"+filename+".csv"):
                continue

            # Loop over the data for each day in the sliding window
            for split in range(VALIDATE_SIZE+TEST_SIZE, TEST_SIZE+1, -1):
                
                # Load data   
                stock_data, stock_id, data_df, lda_news_df, lda_twitter_df, GDELTv1, GDELTv2 = load_data(stock_name)
                
                # Preprocess data
                training_scaled, past_cov_ts, scaler_dataset = preprocess_data(stock_data, data_df, lda_news_df, lda_twitter_df, GDELTv1, GDELTv2, split, features)
                
                # Perform hyperparameter tuning for the current day
                # Predict
                predictions = predict_next_n_days(model, training_scaled, past_cov_ts, scaler_dataset)

                generate_output(filename, predictions, stock_data, stock_id, split, stock_name, generate_path + "/")
            finalize_csv(generate_path+"/"+filename+".csv")        
            #find_best_param(stock_name)
    print(f"Tuning completed for {stock_name}")