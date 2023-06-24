import matplotlib.pyplot as plt
import os
import pandas as pd
import mpl_toolkits.mplot3d  # 
from sklearn import datasets
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,recall_score,precision_score,balanced_accuracy_score
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import librosa
import os
import subprocess

def cast_to_float(elem):
    return float(elem) if elem != '' else 0


def run_saved_model(wlist,blist,input_size, capas, capa_final, activation,
                    activation_final,validation_data,y):

    subprocess.run(['make', 'classification'])


    args = ['./exe', 'input_size:' + str(input_size), 'capas:' + ','.join([str(elem) for elem in capas]), 'capa_final:'+str(capa_final), 'activation:' + ','.join(activation), 'activation_final:' + activation_final,
                            'loss:cross_entropy','loading:1','blist:' + ','.join(blist),'wlist:' + ','.join(wlist),
                            'validation_data:' + ','.join(validation_data),'print:0']

            
    print(' '.join(args))
    out = subprocess.check_output(args)
    # print(out.decode().split('\n'))

    _,_,pred =  out.decode().split('\n')

    pred = [cast_to_float(elem) for elem in pred.split(' ')]

    accuracy = balanced_accuracy_score(y,pred)
    precision = precision_score(y,pred,average='weighted')
    recall_score_ = recall_score(y,pred,average='weighted')
    f1 = f1_score(y,pred,average='weighted')


    print(confusion_matrix(y,pred))

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall score: {recall_score_}')
    print(f'F1 score: {f1}')





wlist = ['./best_result/w_capa_0.csv','./best_result/w_capa_1.csv']
blist = ['./best_result/b_capa_0.csv','./best_result/b_capa_1.csv']



test_file = './datasets/sound_class_test.csv'
y = pd.read_csv(test_file).to_numpy().argmax(axis=1)
validation_data=['./datasets/sound_data_test.csv',test_file]
run_saved_model(wlist,blist,128,[50],24,['sigmoid'],'soft_max',validation_data,y)


