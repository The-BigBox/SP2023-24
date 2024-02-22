import prediction_function as cmd

# for stock in cmd.STOCK_LIST:
#     cmd.stock_tuning(stock, [1])
#     cmd.moving_average(stock)

for stock in cmd.STOCK_LIST:
    cmd.stock_tuning(stock, [1])
    cmd.find_best_param(stock)