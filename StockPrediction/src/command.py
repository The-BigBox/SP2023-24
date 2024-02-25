import prediction_function as cmd
import time

def cal_execution_time(start_time):
    end_time = time.time()
    execution_time = end_time - start_time
    hours = int(execution_time // 3600)
    minutes = int((execution_time % 3600) // 60)
    seconds = int(execution_time % 60)
    print(f"--> Process in {hours} hrs {minutes} min {seconds} sec <--")

start_time = time.time()

# Command Lists
print("-----------------------------------------")
print("1 : Convert stock data to weekly")
print("2 : Tuning moving average")
print("3 : Tuning ARIMA")
print("4 : Tuning machine learning")
print("5 : Find best parameter")
print("-----------------------------------------")
command = input("Enter the command: ")
stock = input("Enter stock name: ").upper()
print("-----------------------------------------")

if command == "1":
    if stock == "ALL":
        for stock in cmd.STOCK_LIST:
            cmd.convert_weekly_data(stock)
    else:
        cmd.convert_weekly_data(stock)

elif command == "2":
    if stock == "ALL":
        for stock in cmd.STOCK_LIST:
            cmd.moving_average(stock)
    else:
        cmd.moving_average(stock)
    
elif command == "3":
    if stock == "ALL":
        for stock in cmd.STOCK_LIST:
            cmd.arima_prediction(stock)
    else:
        cmd.arima_prediction(stock)

elif command == "4":
    print("-> 1 : Fundamental and Technical")
    print("-> 2 : LDA News")
    print("-> 3 : LDA Twitter")
    print("-> 4 : GDELT V1")
    print("-> 5 : GDELT V2")
    print("-----------------------------------------")
    features = input("Enter features list: ")
    features = [int(item.strip()) for item in features.split(',')]
    print("-----------------------------------------")
    if stock == "ALL":
        for stock in cmd.STOCK_LIST:
            cmd.stock_tuning(stock, features)
    else:
        cmd.stock_tuning(stock, features)
    
elif command == "5":
    if stock == "ALL":
        for stock in cmd.STOCK_LIST:
            cmd.find_best_param(stock)
    else:
        cmd.find_best_param(stock)

# cmd.convert_weekly_data(stock)
# cmd.moving_average(stock)
# cmd.stock_tuning(stock, [1])
# cmd.arima_prediction(stock)

# for stock in cmd.STOCK_LIST:

#     cmd.stock_tuning(stock, [1,2,3,4])
#     cmd.find_best_param(stock)

# cmd.stock_tuning("ADVANC", [1,2])
    
cal_execution_time(start_time)
print("-----------------------------------------")