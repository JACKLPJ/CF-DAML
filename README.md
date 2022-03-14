# CF-DAML  
This is an AutoML software based on collaborative filtering.  
1.安装依赖包  
python                    3.8  
xgboost                   1.4.2     
scikit-learn              0.23.2                     
scipy                     1.6.2   
pandas                    1.2.4   
numpy                     1.20.1   

2.下载安装说明  
将整个CF-DAML下载到本机工作盘中（不支持硬盘，U盘）即可。  

3.使用说明  
在CF-DAML文件夹里建立测试文件，比如下图中的test_CF-DAML.ipynb，也可以是.py文件，但要求必须在软件文件夹中测试。  
![image](https://user-images.githubusercontent.com/42956088/158141539-a7b78125-ad44-4468-bf7f-28795375be37.png)


（1）导入函数库  
import pandas as pd  
import numpy as np  
import AutoML8w_stack as automl  
import sklearn  
import time   
（2）软件参数的设置  
CFDAML = automl.Automl(  
                       data_pre_processing = True,# 是否需要数据预处理 (可选：True,False；默认：False)  
                       system = 'linux',# 系统型号（可选：'linux','windows','mac'；默认：'linux'）  
                       N_jobs = -1,# 并行运行的CPU核数 (默认：-1（表示使用机器所有CPU核）)  
                       verbose = False,# 是否显示软件运行的中间结果 (可选：True,False；默认：False)  
                       time_per_model = 360# 训练单个模型管道的时间上限（默认：360（秒））  
                      )   
（3）读取待测数据集并分裂训练集和测试集  
publishing_data = pd.read_csv(  
    '/media/sia1/Elements SE/AutoML测试数据集/DataSets/publishing_data.csv',  
    sep=',',  
    header=None)  
X, y = publishing_data.iloc[:, :9], publishing_data.iloc[:, 9]  
X_train, X_test, y_train, y_test = \  
        sklearn.model_selection.train_test_split(X, y, random_state=42)  
（4）模型的训练和预测  
t0 = time.perf_counter() # 记录训练和测试全过程的时间  
CFDAML.fit(X_train, y_train) # 模型的训练  
y_hat = CFDAML.predict(X_test) # 模型的预测  
print("Runtime: ", time.perf_counter() - t0) # 打印时间开销  
print("Accuracy score: ", sklearn.metrics.accuracy_score(y_test, y_hat)) #打印测试集上的准确率  
（5）结果打印  
Runtime: 22.920272440998815  
Accuracy score: 0.9085545722713865  
