import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # 
from sklearn import datasets
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,recall_score,precision_score,balanced_accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import librosa
import os
import subprocess

def cast_to_float(elem):
    return float(elem) if elem != '' else 0


def run_model(input_size, capas, capa_final, activation, activation_final, loss, epoch, alpha, training_data, validation_data='', compile_=0, print_=0):
    if compile_: 
        subprocess.check_output(['make', 'classification'])


    # out = subprocess.check_output(['./exe'])
    args = ['./exe', 'input_size:' + str(input_size), 'capas:' + ','.join([str(elem) for elem in capas]), 'capa_final:'+str(capa_final), 'activation:' + ','.join(activation), 'activation_final:' + activation_final,
                            'loss:' + loss, 'training_data:' + ','.join(training_data), 'epoch:' + str(epoch), 'alpha:' + str(alpha), 'print:' + str(print_),'validation_data:' + ','.join(validation_data)]
                
            
    # print(' '.join(args))
    out = subprocess.check_output(args)
    # print(out.decode())

    error_train,error_val,pred =  out.decode().split('\n')

    error_train = [cast_to_float(elem)  for elem in error_train.split(' ')]
    error_val = [cast_to_float(elem) for elem in error_val.split(' ')]
    pred = [cast_to_float(elem) for elem in pred.split(' ')]

    # print(pred)

    # print(error_val)
    # print(pred)

    # print(error_train)

    # plt.plot([i for i in range(len(error_train))],error_train)
    # plt.plot([i for i in range(len(error_val))],error_val)
    # plt.show()
    # print(len(pred))

    return pred,error_train,error_val


def save_results(name,y_pred,error_val,error_train,cm):


    np.save(name + '_pred',y_pred)
    np.save(name + '_cm',cm)
    np.save(name + '_error_val',error_val)
    np.save(name + '_error_train',error_train)
    



y = pd.read_csv('./datasets/iris_class_test.csv').to_numpy().argmax(axis=1)

y_pred,error_train,error_val = run_model(4,[5,2],3,['relu','relu'],'sigmoid','mse',1000,0.1,
          ['./datasets/iris_data_train.csv','./datasets/iris_class_train.csv'],
          validation_data=['./datasets/iris_data_test.csv','./datasets/iris_class_test.csv'],
          print_=0)

# print(y.argmax(axis=1))
accuracy = accuracy_score(y,y_pred)
precision = precision_score(y,y_pred,average=None)
recall_score = recall_score(y,y_pred,average=None)
f1 = f1_score(y,y_pred,average=None)

print(accuracy,precision,recall_score,f1)



# run_model(4,[5,2],3,['relu','relu'],'sigmoid','mse',1000,0.1,
#           ['./datasets/iris.csv','./datasets/iris_clases.csv'],
#           print_=1)

