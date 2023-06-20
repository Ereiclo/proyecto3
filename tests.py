import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # 
from sklearn import datasets
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

    out = subprocess.check_output(['./exe', 'input_size:' + str(input_size), 'capas:' + ','.join([str(elem) for elem in capas]), 'capa_final:'+str(capa_final), 'activation:' + ','.join(activation), 'activation_final:' + activation_final,
                            'loss:' + loss, 'training_data:' + ','.join(training_data), 'epoch:' + str(epoch), 'alpha:' + str(alpha), 'print:' + str(print_),]) #'validation_data:' + ','.join(validation_data)])
    error_train,error_val,pred =  out.decode().split('\n')

    error_train = [cast_to_float(elem)  for elem in error_train.split(' ')]
    error_val = [cast_to_float(elem) for elem in error_val.split(' ')]
    pred = [cast_to_float(elem) for elem in pred.split(' ')]


    # print(error_train)

    # plt.plot([i for i in range(len(error_train))],error_train)
    # plt.show()

    return error_train,error_val,pred



run_model(4,[5,2],3,['sigmoid','sigmoid'],'sigmoid','mse',1000,0.1,['./datasets/iris.csv','./datasets/iris_clases.csv'],print_=0)
