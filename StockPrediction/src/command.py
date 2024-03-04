import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import prediction_function as cmd
from datetime import datetime
import time

num_ex = 0
all_features_list = [[1], 
                [1, 2],
                [1, 3],
                [1, 4],
                [1, 5],
                [1, 2, 3],
                [1, 2, 4],
                [1, 2, 5],
                [1, 3, 4],
                [1, 3, 5],
                [1, 4, 5],
                [1, 2, 3, 4],
                [1, 2, 3, 5],
                [1, 2, 4, 5],
                [1, 3, 4, 5],
                [1, 2, 3, 4, 5]]

def cal_execution_time(start_time):
    end_time = time.time()
    execution_time = end_time - start_time
    hours = int(execution_time // 3600)
    minutes = int((execution_time % 3600) // 60)
    seconds = int(execution_time % 60)
    print(f"--> Process in {hours} hrs {minutes} min {seconds} sec <--")
    now = datetime.now()
    print("End date and time:", now)

def get_valid_command():
    print("-----------------------------------------")
    print("1 : Convert stock data to weekly")
    print("2 : Tuning moving average")
    print("3 : Tuning ARIMA")
    print("4 : Tuning machine learning")
    print("5 : Find best parameter")
    print("6 : Merge best parameter")
    print("7 : Get stock change weekly")
    print("-----------------------------------------")
    command = input("Enter the command: ")
    if command not in ["1", "2", "3", "4", "5", "6", "7"]:
        print("Invalid command. Please try again.")
        return get_valid_command()
    return command

def get_valid_stock(command):
    stock = ""
    if command != "6" and command != "7":
        stock = input("Enter stock name (or 'ALL' for all stocks): ").upper()
        print("-----------------------------------------")
        if stock != "ALL" and stock not in cmd.STOCK_LIST:
            print("Stock not found. Please try again.")
            return get_valid_stock()
    return stock

def get_features_list():
    print("-> 1 : Fundamental and Technical")
    print("-> 2 : LDA News")
    print("-> 3 : GDELT V1")
    print("-> 4 : GDELT V2")
    print("-----------------------------------------")
    features = input("Enter features list: ")
    try:
        if features.upper() == "ALL":
            return "ALL"
        features_list = [int(item.strip()) for item in features.split(',')]
    except ValueError:
        print("Invalid input. Please enter numbers separated by commas.")
        return get_features_list()
    return features_list

def execute_command(command, stock, features):
    global num_ex
    if command == "1":
        cmd.convert_weekly_data(stock)
    elif command == "2":
        cmd.moving_average(stock)
    elif command == "3":
        cmd.arima_prediction(stock)
    elif command == "4":
        num_ex += cmd.stock_tuning(stock, features)
    elif command == "5":
        cmd.find_best_param(stock)
    elif command == "6":
        cmd.merge_best_param()
    elif command == "7":
        cmd.get_stock_change()

def main():
    command = get_valid_command()
    stock = get_valid_stock(command)
    features = []
    start_time = time.time()    
    now = datetime.now()
    print("Start date and time:", now)
    print("-----------------------------------------")
    features = get_features_list() if command == "4" else []
    if stock == "ALL":
        for each_stock in cmd.STOCK_LIST:
            if features == "ALL":
                for features in all_features_list:
                    execute_command(command, each_stock, features)
            else:
                execute_command(command, each_stock, features)
        if command == "5":
            cmd.merge_best_param()
    else:
        if features == "ALL":
            for features in all_features_list:
                execute_command(command, stock, features)
        else:
            execute_command(command, stock, features)
    if command == "4":
        print(f"--> Summing number of experiment: {num_ex} <--")
    cal_execution_time(start_time)
    print("-----------------------------------------")

if __name__ == "__main__":
    main()