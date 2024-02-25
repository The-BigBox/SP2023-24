import prediction_function as cmd
import time

num_ex = 0

def cal_execution_time(start_time):
    end_time = time.time()
    execution_time = end_time - start_time
    hours = int(execution_time // 3600)
    minutes = int((execution_time % 3600) // 60)
    seconds = int(execution_time % 60)
    print(f"--> Process in {hours} hrs {minutes} min {seconds} sec <--")

def get_valid_command():
    print("-----------------------------------------")
    print("1 : Convert stock data to weekly")
    print("2 : Tuning moving average")
    print("3 : Tuning ARIMA")
    print("4 : Tuning machine learning")
    print("5 : Find best parameter")
    print("-----------------------------------------")
    command = input("Enter the command: ")
    if command not in ["1", "2", "3", "4", "5"]:
        print("Invalid command. Please try again.")
        return get_valid_command()
    return command

def get_valid_stock():
    stock = input("Enter stock name (or 'ALL' for all stocks): ").upper()
    print("-----------------------------------------")
    if stock != "ALL" and stock not in cmd.STOCK_LIST:
        print("Stock not found. Please try again.")
        return get_valid_stock()
    return stock

def get_features_list():
    print("-> 1 : Fundamental and Technical")
    print("-> 2 : LDA News")
    print("-> 3 : LDA Twitter")
    print("-> 4 : GDELT V1")
    print("-> 5 : GDELT V2")
    print("-----------------------------------------")
    features = input("Enter features list: ")
    try:
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

def main():
    start_time = time.time()
    command = get_valid_command()
    stock = get_valid_stock()
    features = ""
    if command == "4":
        features = get_features_list()
    if stock == "ALL":
        for each_stock in cmd.STOCK_LIST:
            execute_command(command, each_stock, features)
    else:
        execute_command(command, stock)
    if command == "4":
        print(f"--> Summing number of experiment: {num_ex} <--")
    cal_execution_time(start_time)
    print("-----------------------------------------")

if __name__ == "__main__":
    main()