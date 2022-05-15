import pandas as pd
import numpy as np
from sklearn import metrics
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.pyplot as plt


# preprocessing : 股價的漲 (1) or 跌 (0) label
def add_label(df, column):
    label = []
    for i in range(len(df)):
        if(i == 0):
            label.append(0)
            continue
        if(df[column][i] >= df[column][i - 1]):
            label.append(1)
        else:
            label.append(0)
    return pd.Series(label)

# 將股價狀態轉為趨勢分類
def label_convert(train):
    data = []
    for i in range(len(train) - 5):
        if (i == 0):
            data.append(0)
            continue
        if train[i + 5] > train[i] :
            data.append(1)
        else:
            data.append(0)
    return data

if __name__ == '__main__':
    # You should not modify this part.
    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')
    parser.add_argument('--testing',
                        default='testing_data.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()
    
    # The following part is an example.
    # You can modify it at will.

    train = pd.read_csv(args.training,header=None)
    train = train.rename(columns={0:'open',1:'high',2:'low',3:'close'})
    print(train)
    #test = pd.read_csv("test.csv",header=['open','high','low','close'])
    test = pd.read_csv(args.testing,header=None)
    test = test.rename(columns={0:'open',1:'high',2:'low',3:'close'})
    print(test)

    from sklearn.preprocessing import MinMaxScaler 

    train_set = train[['open']]
    test_set = test[['open']]
    ######################################################################
    # preprocessing: 資料正規化
    sc = MinMaxScaler(feature_range = (0, 1))
    train_set= train_set.values.reshape(-1, 1)
    training_set_scaled = sc.fit_transform(train_set)

    train_x, train_y = [], []
    # 以 5 天當作一個單位
    for i in range(5, len(train_set)):
        train_x.append(training_set_scaled[i-5: i, 0]) 
        train_y.append(training_set_scaled[i:i+1, 0]) 
        
    train_x, train_y = np.array(train_x), np.array(train_y) 
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))

    print("finish")
    ######################################################################
    keras.backend.clear_session()
    regressor = Sequential()
    regressor.add(LSTM(units = 50, input_shape = (train_x.shape[1], 1)))
    regressor.add(Dense(units = 1))
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    regressor.summary()
    history = regressor.fit(train_x, train_y, epochs = 40, batch_size = 16)
    print("finish")

    ######################################################################
   
    # loss graph
    plt.title('train_loss')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.plot( history.history["loss"])

    ######################################################################

    dataset_total = pd.concat((train['open'], test['open']), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(test) - 5:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)

    test_x = []
    for i in range(5, len(inputs)):
        test_x.append(inputs[i-5: i, 0])
    test_x = np.array(test_x)
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
    predicted_stock_price = regressor.predict(test_x)
    # 將股價轉為原本的範圍區間
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    # 預測股價趨勢圖
    plt.plot(test['open'].values, color = 'black', label = 'Real')
    plt.plot(predicted_stock_price, color = 'green', label = 'Predicted')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    #plt.show()

    ######################################################################

    ans1 = label_convert(predicted_stock_price)  # 預測的list(5天後)  1:預測5天後會漲 0:預測5天後會跌 
    #print("ans1: "+ str(ans1))
    #ans2 = label_convert(test["open"])
    #print("ans2: "+ str(ans2))

    #print("accuracy :", end=" ")
    #print(str(round(metrics.accuracy_score(ans1, ans2) * 100, 2)) + " %" )
 
    ######################################################################
    ### 輸出 output.csv  策略: 當0轉1的時候，1那個點就要買  當1轉0的時候，0那個點就要賣 (買時股票數量-> 1 賣時股票數量-> -0)
    output=[]
    stock_number=0
    for i in range(len(ans1)-1):
        if i==0:
            if ans1[i]==0:
                output.append(0)
            else:
                output.append(1)
        else:
            if ans1[i]==1 and ans1[i-1]==0:
                output.append(1)
            else: 
                if ans1[i]==0 and ans1[i-1]==1:
                    output.append(-1)
                else:
                    output.append(0)
    
    print('test_set: '+ str(len(test_set)))

    while len(output)<(len(test_set)-1):
        output.append(0)
        
    print(output)

    with open(args.output, 'w') as output_file:
       for i in range(len(output)):
           output_file.write(str(output[i])+'\n')
