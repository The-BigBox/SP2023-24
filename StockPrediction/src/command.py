import prediction_function as cmd
import time

def cal_execution_time(start_time):
    end_time = time.time()
    execution_time = end_time - start_time
    hours = int(execution_time // 3600)
    minutes = int((execution_time % 3600) // 60)
    seconds = int(execution_time % 60)
    print(f"Process in {hours} hrs {minutes} min {seconds} sec")

start_time = time.time()


# cmd.convert_weekly_data(stock)
# cmd.moving_average(stock)
# cmd.stock_tuning(stock, [1])

for stock in cmd.STOCK_LIST:
    cmd.arima_prediction(stock)
    cmd.stock_tuning(stock, [1])
    cmd.stock_tuning(stock, [1,2])
    cmd.stock_tuning(stock, [1,3])
    cmd.find_best_param(stock)

# cmd.stock_tuning("ADVANC", [1,2])
    
cal_execution_time(start_time)
print("-----------------------------------------")