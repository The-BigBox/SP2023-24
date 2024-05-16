import os
import csv
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import ARIMA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from darts.dataprocessing.transformers import Scaler
from darts.models import TiDEModel, BlockRNNModel, XGBModel, LinearRegressionModel, RandomForest

ORIGINAL_DATA_PATH = os.getcwd() + '/data/Fundamental+Technical Data/STOCK_DATA/'
DATA_PATH = os.getcwd() + '/data/Fundamental+Technical Data/STOCK_DATA_WEEKLY/'
LDA_NEWS_1_2_PATH = os.getcwd() + '/data/Online Data/LDA News/News.csv'
LDA_NEWS_3_4_PATH = os.getcwd() + '/data/Online Data/LDA News/News_filtered.csv'
LDA_TWITTER_PATH = os.getcwd() + '/data/Online Data/LDA Twitter/Twitter.csv'
GDELT_V1_PATH = os.getcwd() + '/data/Online Data/GDELT V1/gdelt_v1.csv'
GDELT_V2_PATH = os.getcwd() + '/data/Online Data/GDELT V2/gdelt_v2.csv'
ID_PATH = os.getcwd() + '/data/Fundamental+Technical Data/ID_Name.csv'
PARAMETER_PATH = os.getcwd() + '/model/'
BACKTEST_PATH = os.getcwd() + '/backtest/'
FOCUS_COMPONENT = 'Close'
RETAIN_COMPONENTS = ["Open", "High", "Low", "PE", "PBV", "T_EPS", "FSCORE", "Vol",
                           "Buy Vol", "Sell Vol", "ATO/ATC", "EMA25", "EMA50", "EMA200", "MACD", "RSI"]
TRAINING_SIZE = 139
VALIDATE_SIZE = 35
TEST_SIZE = 35
PREDICT_SIZE = 1
STOCK_LIST = ["ADVANC", "BANPU", "BH", "BTS", "CBG", "CPALL", "CPF", "INTUCH", "IVL", "KBANK", "LH", "PTT", "PTTEP", "PTTGC", "SCB", "SCC", "TISCO", "TU", "WHA"]
MODEL_PARAM = {
    # Tree
    'XGBModel': {
        'lags': [1,2,3,4,8,12,16,20,24], 
        'lags_past_covariates': [1,2,3,4,8,12,16,20,24], 
        'output_chunk_length': [1], 
    },
    'RandomForest': {
        'lags': [1,2,3,4,8,12,16,20,24], 
        'lags_past_covariates': [1,2,3,4,8,12,16,20,24], 
        'output_chunk_length': [1], 
        'n_jobs' : [-1], 
    },

    #Vector
    'LinearRegressionModel': {
        'lags': [1,2,3,4,8,12,16,20,24], 
        'lags_past_covariates': [1,2,3,4,8,12,16,20,24], 
        'output_chunk_length': [1], 
    },

    #RNN
    'BlockRNNModel': {
        'model': ['LSTM'],
        'input_chunk_length': [1,2,3,4,8,12,16,20,24], 
        'output_chunk_length': [1], 
        'n_epochs': [15],#, 50, 100, 150],
    },

    #Transformer
    'TiDEModel': {
        'input_chunk_length': [1,2,3,4,8,12,16,20,24], 
        'output_chunk_length': [1], 
        'n_epochs': [15],
    },  
}

def convert_weekly_data(stock_name): 
    stock_data = pd.read_csv(ORIGINAL_DATA_PATH + stock_name + ".csv")
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    
    # Select every 7th data point from stock data
    stock_data = stock_data.iloc[::7, :]

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH) 
    stock_data.to_csv(DATA_PATH + stock_name + ".csv", index=False)
    print("Converted to weekly for ", stock_name)
      
def load_data(stock_name, news_version):
    stock_data = pd.read_csv(os.path.join(DATA_PATH, (stock_name + ".csv")))
    id_name_map = pd.read_csv(ID_PATH)

    stock_id = id_name_map.loc[id_name_map['Stock_Name'].str.strip() == stock_name, 'Stock_ID'].iloc[0]

    expert_chosen_news_column = ['Date', 'Topic_5', 'Topic_4', 'Topic_38', 'Topic_42', 'Topic_88', 'Topic_90', 'Topic_111', 'Topic_119', 'Topic_128', 'Topic_159', 'Topic_172', 'Topic_173', 'Topic_196', 'Topic_232', 'Topic_240', 'Topic_259', 'Topic_293', 'Topic_294', 'Topic_295', 'Topic_330', 'Topic_373', 'Topic_382']
    if news_version == "1":
        lda_news_df = pd.read_csv(LDA_NEWS_1_2_PATH)
    elif news_version == "2":
        lda_news_df = pd.read_csv(LDA_NEWS_1_2_PATH, usecols=expert_chosen_news_column)
    elif news_version == "3":
        lda_news_df = pd.read_csv(LDA_NEWS_3_4_PATH)
    elif news_version == "4":
        lda_news_df = pd.read_csv(LDA_NEWS_3_4_PATH, usecols=expert_chosen_news_column)

    prefix = "news"
    lda_news_df.columns =  [f"{prefix}_{col}" if col != 'Date' else col for col in lda_news_df.columns]
    
    GDELTv1 = pd.read_csv(GDELT_V1_PATH)
    prefix = "gdeltv1"
    GDELTv1.columns =  [f"{prefix}_{col}" if col != 'Date' else col for col in GDELTv1.columns]
    
    GDELTv2 = pd.read_csv(GDELT_V2_PATH)
    prefix = "gdeltv2"
    GDELTv2.columns =  [f"{prefix}_{col}" if col != 'Date' else col for col in GDELTv2.columns]

    data_df = stock_data.copy()
    return stock_data, stock_id, data_df, lda_news_df, GDELTv1, GDELTv2

def preprocess_online_data(data_df, online_data_df):
    if 'Date' in data_df.columns:
        data_df['Date'] = pd.to_datetime(data_df['Date'])
        data_df.set_index('Date', inplace=True)

    online_data_df['Date'] = pd.to_datetime(online_data_df['Date'])
    online_data_df.set_index('Date', inplace=True)
    
    # Remove news data after the last date in data_df
    last_data_date = data_df.index.max()
    online_data_df = online_data_df[online_data_df.index <= last_data_date]
    
    summed_df = pd.DataFrame(index=data_df.index.unique()).sort_index()
    summed = online_data_df.reindex(summed_df.index, method='ffill').fillna(0)
    summed_df = pd.concat([summed_df, summed], axis=1)
    summed_df.reset_index(inplace=True)

    return summed_df

def preprocess_data(data, data_df,  lda_news_df, GDELTv1, GDELTv2, split, features):
    data = data.dropna()
    serie = data[FOCUS_COMPONENT]
    past_covariate = data[RETAIN_COMPONENTS].apply(pd.to_numeric, errors='coerce').ffill().bfill()
    
    if 2 in features:
        updated_lda_news_df = preprocess_online_data(data_df, lda_news_df)
        past_covariate = past_covariate.join(updated_lda_news_df.reset_index(drop=True).drop(columns='Date'))

    if 3 in features:
        updated_GDELTv1 = preprocess_online_data(data_df, GDELTv1)
        past_covariate = past_covariate.join(updated_GDELTv1.reset_index(drop=True).drop(columns='Date'))

    if 4 in features:
        updated_GDELTv2 = preprocess_online_data(data_df, GDELTv2)
        past_covariate = past_covariate.join(updated_GDELTv2.reset_index(drop=True).drop(columns='Date'))

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
    model.fit(training_scaled, past_covariates=past_cov_ts)
    forecast = model.predict(PREDICT_SIZE)
    in_forecast = scaler_dataset.inverse_transform(forecast)
    return in_forecast

def generate_output(filename, predictions, stock_data, stock_id, split, stock, filepath):
    if split - PREDICT_SIZE <= 0: 
        last_known_date = pd.to_datetime(stock_data['Date'].iloc[-1], dayfirst=True)
        
        if split == 0:
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
    
    combined_df.fillna(0, inplace=True)
    
    output = []
    predict_date = stock_data['Date'].iloc[-split-1]
    
    for _, row in combined_df.iterrows():
        date = row.iloc[0]
        pred_value = round(row.iloc[1], 2)
        output.append([predict_date, stock_id, date, pred_value])
            
    output_df = pd.DataFrame(output, columns=["Predict_Date", "Stock_ID", "Date", "Closing_Price"])
    output_df.to_csv(filepath+filename+".csv", mode='a', header=not os.path.exists(filepath+filename+".csv"))

def finalize_csv(csv_path):
    final_df = pd.read_csv(csv_path)
    
    if 'Unnamed: 0' in final_df.columns:
        final_df.drop('Unnamed: 0', axis=1, inplace=True)
    
    final_df.reset_index(inplace=True)
    final_df.rename(columns={'index': 'Order_ID'}, inplace=True)
    final_df.to_csv(csv_path, index=False)

def arima_prediction(stock_name):
    ARIMA_PARAM = {
        'p': [1,2,3,4,8,12,16,20,24], 
        'd': [1], 
        'q': [1,2,3,4,8,12,16,20,24], 
    }

    if not os.path.exists(PARAMETER_PATH + stock_name):
        os.makedirs(PARAMETER_PATH + stock_name) 
    
    print(f"Tuning {stock_name} with ARIMA ...")
    num_ex = 0
 
    for params in ParameterGrid(ARIMA_PARAM):
        model = ARIMA(**params)

        generate_path = PARAMETER_PATH+f"{stock_name}/ARIMA"
            
        if not os.path.exists(generate_path):
            os.makedirs(generate_path) 

        params_str = '_'.join(f"{key}{val}" for key, val in params.items())
        filename = f"ARIMA_{params_str}"

        if os.path.exists(generate_path+"/"+filename+".csv"):
            continue

        num_ex += 1

        for split in range(VALIDATE_SIZE+TEST_SIZE, TEST_SIZE, -1):
            stock_data, stock_id, data_df, lda_news_df, GDELTv1, GDELTv2 = load_data(stock_name, "1")
            training_scaled, _, scaler_dataset = preprocess_data(stock_data, data_df, lda_news_df, GDELTv1, GDELTv2, split, [1])

            try:
                model.fit(training_scaled)
                forecast = model.predict(PREDICT_SIZE)
                predictions = scaler_dataset.inverse_transform(forecast)
            except Exception as e:
                print(f"Error fitting ARIMA model with params {params}: {e}")

            generate_output(filename, predictions, stock_data, stock_id, split, stock_name, generate_path + "/")
        finalize_csv(generate_path+"/"+filename+".csv")        
    print(f"Completed ARIMA tune for {num_ex} parameters")  
    print("-----------------------------------------")

def moving_average(stock):
    if not os.path.exists(f'{PARAMETER_PATH}/{stock}/Moving Average'):
        os.makedirs(f'{PARAMETER_PATH}/{stock}/Moving Average')

    window_size = [1,2,3,4,8,12,16,20,24]

    for size in window_size:
        stock_data, stock_id, _, _, _, _, _ = load_data(stock, "1")
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

def calculate_directional_accuracy_with_thresholds(actual, forecast):
    thresholds = [0, 5, 10] 
    combinations = [(up, -down) for up in thresholds for down in thresholds if not (up > 0 and down > 0)]
    results = []
    for up, down in combinations:
        acc = 0
        for i in range(len(forecast)):
            actual_change = ((actual[i + 1] - actual[i]) / actual[i]) * 100
            predicted_change = ((forecast[i] - actual[i]) / actual[i]) * 100
            if (actual_change > up and predicted_change > up) or \
                (actual_change < down and predicted_change < down) or\
                (actual_change == predicted_change):
                acc += 1

        da = round(acc / len(forecast) * 100, 2)
        results.append((up, down, da))
    
    return results

def calculate_mape(actual, forecast):
    actual, forecast = np.array(actual), np.array(forecast)
    non_zero_actual = np.where(actual == 0, np.nan, actual)
    mape = np.mean(np.abs((actual - forecast) / non_zero_actual)) * 100
    return np.nanmean(mape)

def calculate_rmse(actual, forecast):
    actual, forecast = np.array(actual), np.array(forecast)
    if len(actual) != len(forecast):
        raise ValueError("The length of actual and forecast arrays must match.")
    mse = np.mean((forecast - actual) ** 2)
    return np.sqrt(mse)

def cal_err_and_acc(predicted_ts, val_ts, condition=True):
    val = val_ts.pd_dataframe()
    predicted = predicted_ts.pd_dataframe()

    val.reset_index(drop=True, inplace=True)
    predicted.reset_index(drop=True, inplace=True)

    actual = val['Close']
    forecast = predicted['Closing_Price']
    
    mape_error = calculate_mape(actual.iloc[1:], forecast)
    rmse_error = calculate_rmse(actual.iloc[1:], forecast)
    dir_acc = calculate_directional_accuracy_with_thresholds(actual, forecast)
    
    if condition:
        return mape_error, rmse_error, dir_acc
    else:
        print(f"MAPE = {mape_error:.2f} %")
        print(f"RMSE = {rmse_error:.2f} %\n")
        print(f"Directional Accuracy = {dir_acc:.2f} %")

def find_best_param(stock_name):
    path = PARAMETER_PATH + stock_name
    overall_best_params = []
    all_dir = os.listdir(path)

    for folder_list in all_dir:
        if folder_list == ".DS_Store" or folder_list == "best_param_overall.csv":
            continue

        best_params_by_mape = {'param': None, 'mape': float('inf'), 'rmse': float('inf'), 'da[0:0]': 0}
        best_params_by_rmse = {'param': None, 'mape': float('inf'), 'rmse': float('inf'), 'da[0:0]': 0}
        best_params_by_dir = {'param': None, 'mape': float('inf'), 'rmse': float('inf'), 'da[0:0]': 0}
        results = []
        
        check_path = path + "/" + folder_list
        dir_list = os.listdir(check_path)
        for param_file in dir_list:
            if param_file == ".DS_Store" or param_file == "all_result.csv" or param_file == "best_parameters.txt":
                continue
            
            predict = pd.read_csv(check_path + '/' + param_file)
            predict_ts = TimeSeries.from_dataframe(predict[['Closing_Price']], time_col=None)
            val = pd.read_csv(DATA_PATH + stock_name + ".csv")
            val = val[['Close']].iloc[-VALIDATE_SIZE-TEST_SIZE-1:-TEST_SIZE]
            val_ts = TimeSeries.from_dataframe(val, time_col=None)    

            try:
                avg_mape, avg_rmse, avg_dir = cal_err_and_acc(predict_ts, val_ts, True)
            except Exception as e:
                print(f"Error {e} when calculate error in {check_path}/{param_file}")
                continue

            results.append({f'param': param_file, 'mape': avg_mape, 'rmse': avg_rmse, **{f'da[{up}:{down}]': da for up, down, da in avg_dir}})

            if avg_dir[0][2] > best_params_by_dir['da[0:0]']:
                best_params_by_dir.update({'param': param_file, 'mape': avg_mape, 'rmse': avg_rmse, **{f'da[{up}:{down}]': da for up, down, da in avg_dir}})
            if avg_mape < best_params_by_mape['mape']:
                best_params_by_mape.update({'param': param_file, 'mape': avg_mape, 'rmse': avg_rmse, **{f'da[{up}:{down}]': da for up, down, da in avg_dir}})
            if avg_rmse < best_params_by_rmse['rmse']:
                best_params_by_rmse.update({'param': param_file, 'mape': avg_mape, 'rmse': avg_rmse, **{f'da[{up}:{down}]': da for up, down, da in avg_dir}})

        results_df = pd.DataFrame(results)
        results_df = results_df.round(2)
        results_df.sort_values(by=['param'], inplace=True)
        results_df.to_csv(check_path + '/all_result.csv', index=False)

        overall_best_params.append({'Features': folder_list, **best_params_by_mape})

        with open(check_path + '/best_parameters.txt', 'w') as f:
            f.write("Best MAPE Param:\n")
            f.write(f"Param: {best_params_by_mape['param']}\n")
            f.write(f"MAPE: {round(best_params_by_mape['mape'], 2)}\n")
            f.write(f"RMSE: {round(best_params_by_mape['rmse'], 2)}\n")
            f.write(f"Dir: {round(best_params_by_mape['da[0:0]'], 2)}\n\n")

            f.write("Best RMSE Param:\n")
            f.write(f"Param: {best_params_by_rmse['param']}\n")
            f.write(f"MAPE: {round(best_params_by_rmse['mape'], 2)}\n")
            f.write(f"RMSE: {round(best_params_by_rmse['rmse'], 2)}\n")  
            f.write(f"Dir: {round(best_params_by_rmse['da[0:0]'], 2)}\n\n")

            f.write("Best Dir Param:\n")
            f.write(f"Param: {best_params_by_dir['param']}\n")
            f.write(f"MAPE: {round(best_params_by_dir['mape'], 2)}\n")
            f.write(f"RMSE: {round(best_params_by_dir['rmse'], 2)}\n")  
            f.write(f"Dir: {round(best_params_by_dir['da[0:0]'], 2)}\n\n")
        
    custom_order = [
        "Moving Average",
        "ARIMA",
        "Fundamental",
        "Fundamental+LDA News",
        "Fundamental+GDELT V1",
        "Fundamental+GDELT V2",
        "Fundamental+LDA News+GDELT V1",
        "Fundamental+LDA News+GDELT V2",
        "Fundamental+GDELT V1+GDELT V2",
        "Fundamental+LDA News+GDELT V1+GDELT V2",
    ]

    overall_best_df = pd.DataFrame(overall_best_params)
    overall_best_df = overall_best_df.round(2)
    order_mapping = {feature: i for i, feature in enumerate(custom_order)}
    overall_best_df['sort_order'] = overall_best_df['Features'].map(order_mapping)
    sorted_df = pd.DataFrame(index=range(len(custom_order))) 
    sorted_df = sorted_df.merge(overall_best_df, left_index=True, right_on='sort_order', how='left')
    sorted_df.drop('sort_order', axis=1, inplace=True)
    sorted_df = sorted_df.astype(object)
    sorted_df.fillna('', inplace=True)
    sorted_df.to_csv(path + '/best_param_overall.csv', index=False)

    print("Finished find best parameter for", stock_name)    
    print("-----------------------------------------")

def merge_best_param():
    base_file_path = PARAMETER_PATH + '{name}/best_param_overall.csv'
    output_file_path = PARAMETER_PATH + 'merged_overall_param.csv'
    best_rows_path = PARAMETER_PATH + 'best_param_with_arima_ma.csv'
    best_ml_path = PARAMETER_PATH + 'best_param_ml.csv'
    best_rows_list = []
    best_ml = []
    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        for stock_name in STOCK_LIST:
            file_path = base_file_path.format(name=stock_name)
            header = f"{stock_name}"
            try:
                df = pd.read_csv(file_path)
                df = df.dropna().reset_index().drop(columns=["index"])
                df.insert(0, 'Stock', stock_name)
                best_rows_list.append(df[df['Features'].str.contains('ARIMA')].copy())
                best_rows_list.append(df[df['Features'].str.contains('Moving Average')].copy())
                filtered_df = df[~((df['Features'] == "ARIMA") | (df['Features'] == "Moving Average"))]
                filtered_row = filtered_df.loc[filtered_df['mape'].idxmin(), 'Features']
                best_ml.append([stock_name, filtered_row])
                best_rows_list.append(df[df['Features'] == filtered_row].copy())
            except Exception as e:
                print(f"Error {e} for {stock_name}, skipping.")
                continue
            writer.writerow([header])
            writer.writerow(df.columns.tolist())
            for index, row in df.iterrows():
                writer.writerow(row)
    best_rows_df = pd.concat(best_rows_list, ignore_index=True)
    best_rows_df.to_csv(best_rows_path, index=False)
    best_ml_df = pd.DataFrame(best_ml, columns=['Stock', 'Best ML Feature'])
    best_ml_df.to_csv(best_ml_path, index=False)
    print(f"Data merged and saved to {output_file_path}")
    print(f"Best rows excluding ARIMA and Moving Average saved to {best_rows_path}")
    print("-----------------------------------------")

def get_stock_change():
    results = []
    for stock in STOCK_LIST:
        data_path = f"{DATA_PATH}/{stock}.csv"
        try:
            data = pd.read_csv(data_path)
        except FileNotFoundError:
            print(f"File not found for {stock}, skipping.")
            continue
        close_prices = data['Close']
        max_diff = max_diff_per = avg = 0
        min_diff = min_diff_per = float('inf') 
        for i in range (len(close_prices)-1):
            diff = abs(close_prices[i+1]-close_prices[i])
            diff_per = (diff/close_prices[i]) * 100
            avg += (diff / close_prices[i]) * 100
            if diff > max_diff:
                max_diff = diff
            if diff < min_diff and diff != 0:
                min_diff = diff
            if diff_per > max_diff_per:
                max_diff_per = diff_per
            if diff < min_diff_per and diff_per != 0:
                min_diff_per = diff_per
        avg_diff = avg/(len(close_prices)-1)
        results.append([stock, round(max_diff, 2), round(min_diff, 2), round(avg_diff, 2), round(max_diff_per, 2), round(min_diff_per, 2)])

    results_df = pd.DataFrame(results, columns=["Stock", "Max(Price)", "Min(Price)", "Average(%)", "Max(%)", "Min(%)"])
    output_path = f"{DATA_PATH}../stock_price_changes_summary.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
  
def stock_tuning(stock_name, features, model_list, news_version):
    if not os.path.exists(PARAMETER_PATH + stock_name):
        os.makedirs(PARAMETER_PATH + stock_name) 

    num_ex = 0
    
    print(f"Tuning {stock_name} with {features} ...")
 
    for model_type, params_grid in MODEL_PARAM.items():
        print("Model: ", model_type)
        for params in ParameterGrid(params_grid):
            if model_type == 'XGBModel' and 1 in model_list:
                model = XGBModel(**params)
            elif model_type == 'RandomForest' and 2 in model_list:
                model = RandomForest(**params)
            elif model_type == 'LinearRegressionModel' and 3 in model_list:
                model = LinearRegressionModel(**params)
            elif model_type == 'BlockRNNModel' and 4 in model_list:
                model = BlockRNNModel(**params)
            elif model_type == 'TiDEModel' and 5 in model_list:
                model = TiDEModel(**params)
            else:
                continue

            generate_path = PARAMETER_PATH+f"{stock_name}/"
            if 1 in features:
                generate_path = generate_path+"Fundamental"
            if 2 in features:
                if news_version == "1":
                    generate_path = generate_path+"+LDA News"
                elif news_version == "2":
                    generate_path = generate_path+"+LDA News(Related Columns)"
                elif news_version == "3":
                    generate_path = generate_path+"+LDA News(Related News)"
                elif news_version == "4":
                    generate_path = generate_path+"+LDA News(Related News and Columns)"
            if 3 in features:
                generate_path = generate_path+"+GDELT V1"
            if 4 in features:
                generate_path = generate_path+"+GDELT V2"
                
            if not os.path.exists(generate_path):
                os.makedirs(generate_path) 

            params_for_filename = {k: v for k, v in params.items() if k != 'n_jobs'}
            params_str = '_'.join(f"{key}{val}" for key, val in params_for_filename.items())
            filename = f"{model_type}_{params_str}"

            if os.path.exists(generate_path+"/"+filename+".csv"):
                continue

            num_ex += 1
            for split in range(VALIDATE_SIZE+TEST_SIZE, TEST_SIZE, -1):
                stock_data, stock_id, data_df, lda_news_df, GDELTv1, GDELTv2 = load_data(stock_name, news_version)
                training_scaled, past_cov_ts, scaler_dataset = preprocess_data(stock_data, data_df, lda_news_df, GDELTv1, GDELTv2, split, features)
                predictions = predict_next_n_days(model, training_scaled, past_cov_ts, scaler_dataset)
                generate_output(filename, predictions, stock_data, stock_id, split, stock_name, generate_path + "/")
            finalize_csv(generate_path+"/"+filename+".csv") 
    print(f"Completed tune for {num_ex} parameters")       
    print("-----------------------------------------")
    return num_ex

def gen_test_for_backtest(stock_name):
    if not os.path.exists(BACKTEST_PATH + stock_name):
        os.makedirs(BACKTEST_PATH + stock_name) 
    generate_path =  +f"{stock_name}/"

    # Insight Wave
    filename = "InsightWave"
    if os.path.exists(generate_path+"/"+filename+".csv"):
        print(f"Backtest of {filename} for {stock_name} is already done.")
    
    else:
        print(f"Generating Backtest {filename} for ",stock_name)
        print("-> 1 : XGBoost")
        print("-> 2 : RandomForest")
        print("-> 3 : LinearRegression")
        print("-> 4 : BlockRNN")
        print("-> 5 : TiDE")
        print("-----------------------------------------")
        model_type = input("Enter model type: ")

        if model_type == '1':
            lags = int(input("Enter lags: "))
            lags_past_covariates = int(input("Enter lags_past_covariates: "))
            model = XGBModel(lags = lags, lags_past_covariates = lags_past_covariates, output_chunk_length = 1)
        elif model_type == '2':
            lags = int(input("Enter lags: "))
            lags_past_covariates = int(input("Enter lags_past_covariates: "))
            model = RandomForest(lags = lags, lags_past_covariates = lags_past_covariates, output_chunk_length = 1, n_jobs = -1)
        elif model_type == '3':
            lags = int(input("Enter lags: "))
            lags_past_covariates = int(input("Enter lags_past_covariates: "))
            model = LinearRegressionModel(lags = lags, lags_past_covariates = lags_past_covariates, output_chunk_length = 1)
        elif model_type == '4':
            input_chunk_length = int(input("Enter input_chunk_length: "))
            model = BlockRNNModel(model = 'LSTM', input_chunk_length = input_chunk_length, output_chunk_length = 1, n_epochs = 15)
        elif model_type == '5':
            input_chunk_length = int(input("Enter input_chunk_length: "))
            model = TiDEModel(input_chunk_length = input_chunk_length, output_chunk_length = 1, n_epochs = 15)
        else:
            print("Error model not found.")
            return

        print("-> 1 : Fundamental and Technical")
        print("-> 2 : LDA News")
        print("-> 3 : GDELT V1")
        print("-> 4 : GDELT V2")
        print("-----------------------------------------")
        features = input("Enter features list: ")
        features_list = [int(item.strip()) for item in features.split(',')]

        if "2" in features_list:
            print("-> 1 : All News")
            print("-> 2 : Filtered News from Econ Topic from Topic Distribution.")
            print("-> 3 : Filtered News from Econ News from expert.")
            print("-> 4 : Filtered News from Econ News and Econ Topic.")
            print("-----------------------------------------")
            news_version = input("Enter news version: ")
            if news_version not in ["1", "2", "3", "4"]:
                print("Invalid input. Please enter only number.")
                return
        else:
            news_version = "1"

        for split in range(TEST_SIZE, 0, -1):
            stock_data, stock_id, data_df, lda_news_df, GDELTv1, GDELTv2 = load_data(stock_name, news_version)
            training_scaled, past_cov_ts, scaler_dataset = preprocess_data(stock_data, data_df, lda_news_df, GDELTv1, GDELTv2, split, features_list)
            predictions = predict_next_n_days(model, training_scaled, past_cov_ts, scaler_dataset)
            generate_output(filename, predictions, stock_data, stock_id, split, stock_name, generate_path + "/")
        finalize_csv(generate_path+"/"+filename+".csv")
    
    # ARIMA
    filename = "ARIMA"
    if os.path.exists(generate_path+"/"+filename+".csv"):
        print(f"Backtest of {filename} for {stock_name} is already done.")
    
    else:
        print("Generate Backtest ARIMA for",stock_name)
        p = int(input("Enter p value: "))
        d = int(input("Enter d value: "))
        q = int(input("Enter q value: "))
 
    
        model = ARIMA(p = p, d = d, q = q) 

        for split in range(TEST_SIZE, 0, -1):
            stock_data, stock_id, data_df, lda_news_df, lda_twitter_df, GDELTv1, GDELTv2 = load_data(stock_name)
            training_scaled, _, scaler_dataset = preprocess_data(stock_data, data_df, lda_news_df, lda_twitter_df, GDELTv1, GDELTv2, split, [1])

            try:
                model.fit(training_scaled)
                forecast = model.predict(PREDICT_SIZE)
                predictions = scaler_dataset.inverse_transform(forecast)
            except Exception as e:
                print(f"Error fitting ARIMA model: {e}")

            generate_output(filename, predictions, stock_data, stock_id, split, stock_name, generate_path + "/")
        finalize_csv(generate_path+"/"+filename+".csv")   
    
    print("-----------------------------------------")

def get_sim_fri(starting_money, stock, mode, percentage):

    log_rows = []

    # Buy in Closing Price in Friday
    val = pd.read_csv(f"{DATA_PATH}/{stock}.csv")
    val = val[['Date','Open','Close']].iloc[-TEST_SIZE-1:].reset_index(drop=True)
    predict = pd.read_csv(f"{BACKTEST_PATH}/{stock}/{mode}.csv")
    predict = predict[['Date', 'Closing_Price']]

    # Backtest parameters
    stock_count = 0
    money = starting_money
    buying_price = val.loc[0]['Close']

    for i in range(len(predict)):
        act = val.loc[i]['Close']
        pred = predict.loc[i]['Closing_Price']
        date = val.loc[i]['Date']
        
        diff = ((pred - act) / act) * 100
        cut_lost = ((act - buying_price) / buying_price) * 100

        if diff > percentage:
            # Assuming you buy as much stock as possible with the money you have
            bought_stocks = money // act
            bought_stocks = int(bought_stocks / 100) * 100
            if money >= bought_stocks * act:
                money -= bought_stocks * act
                stock_count += bought_stocks
                buying_price = act
                action = "Buy"
            else:
                action = "Hold"
        elif diff < -percentage and stock_count > 0:
            # Sell all stocks
            money += stock_count * act
            action = "Sell"
            stock_count = 0
        elif cut_lost < -percentage and stock_count > 0:
            # Sell all stocks
            money += stock_count * act
            action = "Stop loss"
            stock_count = 0
        else:
            action = "Hold"
        
        log_rows.append({
            "Date": date,
            "Action": action,
            "Stocks Count": stock_count,
            "Stock Price": act,
            "Predicted Price": pred,
            "Percent Change": round(diff,2),
            "Money Left": money
        })

    # Sell any remaining stocks at the last available price
    final_money = money + stock_count * val.iloc[-1]['Close']
    profit_loss = final_money - starting_money

    final_money_display = f"{final_money} ({'+' if profit_loss > 0 else ''}{profit_loss})"

    log_rows.append({
        "Date": "SUMMARY",
        "Action": "Final Result",
        "Stocks Count": stock_count,
        "Stock Price": "N/A",
        "Predicted Price": "N/A",
        "Percent Change": "N/A",
        "Money Left": final_money_display
    })

    df = pd.DataFrame(log_rows)
    df.to_csv(f'{BACKTEST_PATH}/{stock}/{stock} {mode} Friday.csv', index=False)

def get_sim_mon(starting_money, stock, mode, percentage):

    log_rows = []

    # Buy in Closing Price in Monday
    val = pd.read_csv(f"{DATA_PATH}/{stock}.csv")
    val = val[['Date','Open','Close']].iloc[-TEST_SIZE-1:].reset_index(drop=True)
    predict = pd.read_csv(f"{BACKTEST_PATH}/{stock}/{mode}.csv")
    predict = predict[['Date', 'Closing_Price']]

    open_data = pd.read_csv(f"{ORIGINAL_DATA_PATH}/{stock}.csv")

    # Backtest parameters
    
    stock_count = 0
    money = starting_money
    buying_price = val.loc[0]['Close']

    for i in range(len(predict)):
        specific_date_from_val = val.loc[i]['Date']
        today_row = open_data.index[open_data['Date'] == specific_date_from_val].to_list()[0]
        next_open_value = open_data.loc[today_row+1]['Open']
        next_date = open_data.loc[today_row+1]['Date']
    
        act = val.loc[i]['Close']
        pred = predict.loc[i]['Closing_Price']
        date = next_date

        diff = ((pred - act) / act) * 100
        cut_lost = ((act - buying_price) / buying_price) * 100

        if diff > percentage:
            # Assuming you buy as much stock as possible with the money you have
            bought_stocks = money // next_open_value
            bought_stocks = int(bought_stocks / 100) * 100
            if money >= bought_stocks * next_open_value:
                money -= bought_stocks * next_open_value
                stock_count += bought_stocks
                buying_price = next_open_value
                action = "Buy"
            else:
                action = "Hold"
        elif diff < -percentage and stock_count > 0:
            # Sell all stocks
            money += stock_count * next_open_value
            action = "Sell"
            stock_count = 0
        elif cut_lost < -percentage and stock_count > 0:
            # Sell all stocks
            money += stock_count * next_open_value
            action = "Stop loss"
            stock_count = 0
        else:
            action = "Hold"

        log_rows.append({
            "Date": date,
            "Action": action,
            "Stocks Count": stock_count,
            "Stock Price": next_open_value,
            "Predicted Price": pred,
            "Percent Change": round(diff,2),
            "Money Left": money
        })

    # Sell any remaining stocks at the last available price
    final_money = money + stock_count * val.iloc[-1]['Close']
    profit_loss = final_money - starting_money
    final_money_display = f"{final_money} ({'+' if profit_loss > 0 else ''}{profit_loss})"

    log_rows.append({
        "Date": "SUMMARY",
        "Action": "Final Result",
        "Stocks Count": stock_count,
        "Stock Price": "N/A",
        "Predicted Price": "N/A",
        "Percent Change": "N/A",
        "Money Left": final_money_display
    })

    df = pd.DataFrame(log_rows)
    df.to_csv(f'{BACKTEST_PATH}/{stock}/{stock} {mode} Monday.csv', index=False)

def cal_backtest(stock):
    percentage = 5
    starting_money = 1000000
    get_sim_mon(starting_money, stock, "ARIMA", percentage)
    get_sim_mon(starting_money, stock, "InsightWave", percentage)
    get_sim_fri(starting_money, stock, "ARIMA", percentage)
    get_sim_fri(starting_money, stock, "InsightWave", percentage)
