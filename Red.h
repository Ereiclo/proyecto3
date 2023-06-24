#include <iostream>
#include <algorithm>
#include <fstream>
#include <armadillo>
#include <deque>
#include <vector>
#include <functional>

using namespace std;

//definicion de una funcion de activación
struct activation_function
{
    function<arma::Row<double>(arma::Row<double>)> f;
    function<arma::Row<double>(arma::Row<double>)> df;
    string name;
};

//definicion de una funcion de loss
struct loss_function
{

    function<double(arma::Row<double>, arma::Row<double>)> loss;
    function<arma::Row<double>(arma::Row<double>, arma::Row<double>)> dloss;
    string name;
};

class Red
{

    //matrices para cada capa
    vector<arma::Mat<double>> w;
    //bias para cada capa
    vector<arma::Row<double>> bias;
    //activacion para cada capa
    vector<activation_function> activation;
    //funcion de loss
    loss_function floss;
    double alpha = 0.15;

public:
    Red(int input_n, vector<int> &n_por_capas, int output_n, vector<activation_function> activation,
        activation_function activation_final, loss_function loss,
        double alpha = 0.15) : activation{activation}, floss{loss}, alpha{alpha}
    {

        if(activation_final.name == "soft_max" && loss.name != "cross_entropy")
            throw std::invalid_argument("Soft max solo se puede usar con cross entropy");

        this->activation.push_back(activation_final);
        w.push_back((arma::Mat<double> (input_n, n_por_capas[0], arma::fill::randu) - 0.5)*0.6);
        bias.push_back(arma::Row<double>(n_por_capas[0], arma::fill::zeros));

        //por cada capa crear la matriz correspondiente
        for (int i = 1; i < n_por_capas.size(); ++i)
        {
            int n = n_por_capas[i - 1];
            int m = n_por_capas[i];
            w.push_back( (arma::Mat<double>(n, m, arma::fill::randu) - 0.5)*0.6);
            bias.push_back(arma::Row<double>(m, arma::fill::zeros));
            



        }

        w.push_back( (arma::Mat<double>(n_por_capas.back(), output_n, arma::fill::randu) - 0.5)*0.6);
        bias.push_back(arma::Row<double>(output_n, arma::fill::zeros));
    }

    void print_red()
    {

        //imprimir matrices de pesos 
        for (int i = 0; i < w.size(); ++i)
        {
            auto m_capa_actual = w[i];
            auto bias_actual = bias[i];

            cout << m_capa_actual << endl;
            cout << bias_actual;
        }
    }



    //definiciones staticas para cada funcion de activacion
    static activation_function sigmoid()
    {
        auto f = [](arma::Row<double> net) -> arma::Row<double>
        {
            return 1.0 / (1.0 + arma::exp(-net));
        };
        auto df = [f](arma::Row<double> net) -> arma::Row<double>
        {
            return f(net) % (1.0 - f(net));
        };

        return activation_function{f, df, "sigmoid"};
    }

    static activation_function tanh()
    {
        auto f = [](arma::Row<double> net) -> arma::Row<double>
        {
            return 2.0 / (1.0 + exp(-2.0 * net)) - 1.0;
        };
        auto df = [f](arma::Row<double> net) -> arma::Row<double>
        {
            return 1.0 - arma::square(f(net));
        };

        return activation_function{f, df, "tanh"};
    }

    static activation_function relu()
    {
        auto f = [](arma::Row<double> net) -> arma::Row<double>
        {
            return arma::max(net, arma::Row<double>(size(net), arma::fill::zeros));
        };
        auto df = []( arma::Row<double> net) -> arma::Row<double>
        {
            arma::Row<arma::uword> result = (net >= arma::Row<double>(size(net), arma::fill::zeros));


            return arma::conv_to<arma::Row<double>>::from(result);
        };

        return activation_function{f, df, "relu"};
    }

    static activation_function soft_max()
    {
        auto f = [](arma::Row<double> net) -> arma::Row<double>
        {
            return arma::exp(net) / (arma::sum(arma::exp(net)));
        };
        auto df = [](arma::Row<double> X) -> arma::Row<double>
        {
            return {};
        };

        return activation_function{f, df, "soft_max"};
    }

    //definiciones estáticas para cada funcionde perdida

    static loss_function mse_loss()
    {

        auto loss = [](arma::Row<double> Y_pred, arma::Row<double> Y) -> double
        {
            return arma::accu(arma::square(Y_pred - Y) / size(Y_pred).n_cols);
        };

        auto dloss = [](arma::Row<double> Y_pred, arma::Row<double> Y) -> arma::Row<double>
        {
            return (Y_pred - Y)*(2.0/size(Y_pred).n_cols);
        };

        return loss_function{loss, dloss, "mse"};
    }

    static loss_function cross_entropy_loss()
    {

        auto loss = [](arma::Row<double> Y_pred, arma::Row<double> Y) -> double
        {
            return (-1.0) * arma::sum(Y % arma::log2(Y_pred));
        };

        auto dloss = [](arma::Row<double> Y_pred, arma::Row<double> Y) -> arma::Row<double>
        {
            double eps = 0.0000000000000001;

            return (-1.0) * (Y / (Y_pred + eps));
        };

        return loss_function{loss, dloss, "cross_entropy"};
    }


    

    //backprogation
    void backpropagate(arma::Row<double> &X, arma::Row<double> &Y,int thrw)
    {

        arma::Row<double> actual = X;
        vector<arma::Row<double>> net_by_layers;

        //guardar el net para cada capa
        for (int i = 0; i < w.size(); ++i)
        {
            arma::Row<double> net = actual * w[i] + bias[i];
            actual = activation[i].f(net);

            net_by_layers.push_back(net);
        }


        deque<arma::Mat<double>> w_derivates;
        deque<arma::Row<double>> bias_derivates;
        arma::Row<double> ro; //dL/dNetj

        for (int current_layer = net_by_layers.size() - 1; current_layer >= 0; --current_layer)
        {
            arma::Row<double> net = net_by_layers[current_layer];
            arma::Row<double> sj = activation[current_layer].f(net); //f(net)
            //dNet/dwj (es siempre el input de la capa anterior o sj^(h-1))
            arma::Row<double> net_wj = current_layer == 0 ? X : activation[current_layer - 1].f(net_by_layers[current_layer - 1]);




            if (current_layer == net_by_layers.size() - 1)
            {

                
                //dsj/dNetj
                arma::Row<double> dsj_Netj = activation[current_layer].df(net);

                if (floss.name == "cross_entropy" && activation[current_layer].name == "soft_max")
                {
                    ro = (sj - Y);
                }
                else
                    ro = floss.dloss(sj, Y) % dsj_Netj;
                    //dL/dNetj
                

                                        //dL/dNetj*dNetj/dwj
                w_derivates.push_front(net_wj.t() * ro);
                //ro = dL/dNetj = dL/dNetj = dL/dNetj * 1 = dL/dNetj * dNetj/db
                bias_derivates.push_front(ro);
            }
            else
            {
                //dNeth+1/dsj
                arma::Mat<double> net_next_h_sj = w[current_layer + 1];

                //dsj/dNetj
                arma::Row<double> dsj_Netj = activation[current_layer].df(net);
                arma::Row<double> temp = (net_next_h_sj * (ro.t())).t();//(dNeth+1/dsj^h)*(dL/dNet^h+1)
                ro = temp % (dsj_Netj); //(dL/dsj^h)*(dsj^h/dNetj)



                                        //dL/dNetj*dNetj/dwj
                w_derivates.push_front(net_wj.t() * ro);
                bias_derivates.push_front(ro);
            }

        }


        //actualizar las derivadas en cada capa
        for (int i = 0; i < w.size(); ++i)
        {
            w[i] = w[i] - alpha * w_derivates[i];
            bias[i] = bias[i] - alpha * bias_derivates[i];
        }
    }

    pair<vector<double>, vector<double>> train(arma::Mat<double> &X_train, arma::Mat<double> &Y_train, int epoch, int print = 1, arma::Mat<double> X_validation = {}, arma::Mat<double> Y_validation = {})
    {
        vector<double> loss_training;
        vector<double> loss_validation;
        //orden en el que se va a acceder el dataset de train
        vector<int> indexes(size(X_train).n_rows);

        std::iota(indexes.begin(),indexes.end(),0);

        for (int i = 0; i < epoch; ++i)
        {
            //desordenar el acceso al dataset de train
            random_shuffle(indexes.begin(),indexes.end());
            double actual_error_train = 0;
            double actual_error_val = 0;
            for (int r = 0; r < indexes.size(); ++r)
            {

                int row = indexes[r];
                arma::Row<double> actual_X = X_train.row(row);
                arma::Row<double> actual_Y = Y_train.row(row);


                //guardar el error actual
                actual_error_train += floss.loss(pred(actual_X), actual_Y);
                backpropagate(actual_X, actual_Y,0);
            }

            for (int row = 0; row < size(X_validation).n_rows; ++row)
            {
                arma::Row<double> actual_X = X_validation.row(row);
                arma::Row<double> actual_Y = Y_validation.row(row);

                //guardar el error actual de validation
                actual_error_val += floss.loss(pred(actual_X), actual_Y);
            }

            loss_training.push_back(actual_error_train);
            if(size(X_validation).n_rows)
                loss_validation.push_back(actual_error_val);

            if (print)
            {
                cout << "El error para la epoca " << i << " es " << actual_error_train << " " <<  actual_error_val << endl;
            }
        }
        return {loss_training, loss_validation};
    }



    

    //predicir el punto actual
    arma::Row<double> pred(arma::Row<double> &X)
    {
        
        arma::Row<double> actual = X;

        for (int i = 0; i < w.size(); ++i)
        {
            actual = activation[i].f(actual * w[i] + bias[i]);
        }

        return actual;
    }

    void save_red(string base_dir){
        
        for(int i = 0; i < w.size();++i){

            w[i].save(base_dir + "w_capa_" + to_string(i) + ".csv",arma::csv_ascii);
            bias[i].save(base_dir + "b_capa_" + to_string(i) + ".csv",arma::csv_ascii);


        }
        
    }


    void load_red(vector<string> wlist,vector<string> blist){

        for(int i= 0; i < w.size();++i){
            w[i].load(wlist[i]);
            bias[i].load(blist[i]);
        }


    }

    void show_red(){

        for(int i= 0; i < w.size();++i){
            cout<<size(w[i])<<endl;
            cout<<size(bias[i])<<endl;
        }


    }
};



//utilidades para leer csv y guardar 

vector<string> readLineStrings(string line)
{

    stringstream ss(line);
    string item;
    vector<string> result;

    // separar la string line por ','
    while (getline(ss, item, ','))
    {
        item.erase(remove(item.begin(), item.end(), ' '), item.end());
        result.push_back(item);
    }

    return result;
}

vector<double> readLineNumbers(string line)
{

    stringstream ss(line);
    string item;
    vector<double> result;

    // separar la string line por ','
    while (getline(ss, item, ','))
    {
        item.erase(remove(item.begin(), item.end(), ' '), item.end());
        result.push_back(stod(item));
    }

    return result;
}

arma::Mat<double> read_data(string name)
{

    if(name == "") return {};
    fstream file;
    string buffer;
    arma::Mat<double> matrix;

    file.open(name, ios::in);


    getline(file, buffer, '\n');

    while (getline(file, buffer, '\n'))
    {

        arma::Row<double> actual_data(readLineNumbers(buffer));

        if (matrix.is_empty())
        {
            matrix = actual_data;
        }
        else
        {
            matrix = arma::join_cols(matrix, actual_data);
        }
    }

    file.close();

    return matrix;
}