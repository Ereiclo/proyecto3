import matplotlib.pyplot as plt
import os
import pandas as pd
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


def run_model(input_size, capas, capa_final, activation, activation_final, loss, epoch, alpha, training_data, validation_data='', compile_=0, print_=0,base_dir=''):
    if compile_: 
        subprocess.check_output(['make', 'classification'])


    # out = subprocess.check_output(['./exe'])
    args = ['./exe', 'input_size:' + str(input_size), 'capas:' + ','.join([str(elem) for elem in capas]), 'capa_final:'+str(capa_final), 'activation:' + ','.join(activation), 'activation_final:' + activation_final,
                            'loss:' + loss, 'training_data:' + ','.join(training_data), 'epoch:' + str(epoch), 'alpha:' + str(alpha), 'print:' + str(print_),'validation_data:' + ','.join(validation_data),'base_dir:' + base_dir]
                
            
    print(' '.join(args))
    out = subprocess.check_output(args)
    # print(out.decode())

    error_train,error_val,pred =  out.decode().split('\n')

    error_train = [cast_to_float(elem)  for elem in error_train.split(' ')]
    error_val = [cast_to_float(elem) for elem in error_val.split(' ')]
    pred = [cast_to_float(elem) for elem in pred.split(' ')]

    # print(len(pred))

    # print(error_train)
    # print(error_val)
    # print(pred)

    # print(error_train)

    # plt.plot([i for i in range(len(error_train))],error_train)
    # plt.plot([i for i in range(len(error_val))],error_val)
    # plt.show()
    # print(len(pred))

    return pred,error_train,error_val


def save_results(y_pred,error_val,error_train,cm,path, name = ''):


    np.save(path + name + 'pred',y_pred)
    np.save(path + name + 'cm',cm)
    np.save(path + name + 'error_val',error_val)
    np.save(path + name + 'error_train',error_train)

    plt.plot([i for i in range(len(error_train))],error_train)
    plt.plot([i for i in range(len(error_val))],error_val)
    plt.savefig(path + name + 'error_fig')
    plt.clf()


    confusion_matrix_file = open(path + name + 'confusion_matrix.txt','w')
    confusion_matrix_file.write(str(cm))



def run_tests():
    activation_function = ['sigmoid','tanh','relu']
    error = 'cross_entropy'
    base_path = './tests/'
    final_activation = 'soft_max'
    alpha = [0.0001,0.001,0.01,0.1,0.15]
    capas_up = [50,100,200]
    capas_down = [200,100,50]
    input_size = 128
    epochs = 200 
    output_size = 24
    max_capas = 3
    training_data = ['./datasets/sound_data_train.csv','./datasets/sound_class_train.csv']
    test_file = './datasets/sound_class_test.csv'
    validation_data=['./datasets/sound_data_test.csv',test_file]
    y = pd.read_csv(test_file).to_numpy().argmax(axis=1)

    total = len(activation_function)*(max_capas-1)*2*len(alpha) +  len(activation_function)*len(alpha) - 1
    # total = len(activation_function)*(max_capas-1)*2*len(alpha) +  len(activation_function)*len(alpha)
    print(total)


    results_dataframe = pd.DataFrame(columns=['activation_name','capa_1','capa_2','capa_3','alpha','accuracy','precision','recall_score','f1'])
    i = 0

    for activation in activation_function:
        for capas in range(max_capas):

            activation_reps = [activation for _ in range(capas+1)]
            cps = [capas_up[:capas+1],capas_down[:capas+1]] if capas > 0 else [capas_up[:1]]
            for cp in cps:
                for alp in alpha:

                    print(f'Test {i} de {total}:')

                    i += 1


                    dir_name = f'{activation}_{"_".join([(str(cp[j]) if j < len(cp) else "") for j in range(max_capas) ])}_{alp}/'

                    if not os.path.exists(base_path + dir_name):
                        os.mkdir(base_path + dir_name)

                    y_pred,error_train,error_val = run_model(input_size,cp,output_size,activation_reps,final_activation,error,
                          epochs,alp,training_data,validation_data,base_dir=base_path+dir_name)


                    # print(dir_name)


                    accuracy = balanced_accuracy_score(y,y_pred)
                    precision = precision_score(y,y_pred,average='weighted')
                    recall_score_ = recall_score(y,y_pred,average='weighted')
                    f1 = f1_score(y,y_pred,average='weighted')

                    # csv_entry = f'{activation},{",".join([(str(cp[j]) if j < len(cp) else "") for j in range(max_capas) ])},{alp},{accuracy},{precision},{recall_score_},{f1}'
                    # print(csv_entry.split(','))

                    new_row = pd.DataFrame({'activation_name':activation,'capa_1': cp[0],
                                  'capa_2': cp[1] if 1 < len(cp) else '',
                                  'capa_3': cp[2] if 2 < len(cp) else '','alpha':alp,'accuracy':accuracy,'precision':precision,
                                  'recall_score':recall_score_,'f1':f1},index=[0])

                    results_dataframe = pd.concat([results_dataframe,new_row])



                    save_results(y_pred,error_val,error_train,confusion_matrix(y,y_pred),base_path + dir_name)


                    
                    




                    # print(accuracy,precision,recall_score,f1)
                    # print(confusion_matrix(y,y_pred))

    
    results_dataframe.to_csv(base_path + 'results.csv',index=False)
 



# run_tests()

# training_data = ['./datasets/sound_data_train.csv','./datasets/sound_class_train.csv']
# test_file = './datasets/sound_class_test.csv'
# validation_data=['./datasets/sound_data_test.csv',test_file]
# y = pd.read_csv(test_file).to_numpy().argmax(axis=1)



# run_model(128,[50],128,['sigmoid'],'soft_max','cross_entropy',200,0.15)