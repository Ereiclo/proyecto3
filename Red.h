#include <iostream>
#include <algorithm>
#include <fstream>
#include <armadillo>
#include <deque>
#include <vector>
#include <functional>

using namespace std;

struct activation_function
{
    function<arma::Row<double>(arma::Row<double>)> f;
    function<arma::Row<double>(arma::Row<double>)> df;
    string name;
};

struct loss_function
{

    function<double(arma::Row<double>, arma::Row<double>)> loss;
    function<arma::Row<double>(arma::Row<double>, arma::Row<double>)> dloss;
    string name;
};

class Red
{

    vector<arma::Mat<double>> w;
    vector<arma::Row<double>> bias;
    vector<activation_function> activation;
    loss_function floss;
    // vector<arma::Mat<double>> f;
    double alpha = 0.15;

public:
    Red(int input_n, vector<int> &n_por_capas, int output_n, vector<activation_function> activation,
        activation_function activation_final, loss_function loss,
        double alpha = 0.15) : activation{activation}, floss{loss}, alpha{alpha}
    {

        // if(activation_final.name == "soft_max" && loss.name != "cross_entropy")
        // throw std::invalid_argument("Soft max solo se puede usar con cross entropy");

        this->activation.push_back(activation_final);
        // this->capas[0](3,3);
        w.push_back((arma::Mat<double> (input_n, n_por_capas[0], arma::fill::randu) - 0.5)*0.6);
        bias.push_back(arma::Row<double>(n_por_capas[0], arma::fill::zeros));

        for (int i = 1; i < n_por_capas.size(); ++i)
        {
            int n = n_por_capas[i - 1];
            int m = n_por_capas[i];
            w.push_back( (arma::Mat<double>(n, m, arma::fill::randu) - 0.5)*0.6);
            bias.push_back(arma::Row<double>(m, arma::fill::zeros));
            

            // for(int ip = 0; ip < n;++ip){
            //     for(int jp = 0; jp  < m ;++jp){
            //         w.back()(ip,jp) = 0.1;
            //     }
            // }



        }

        w.push_back( (arma::Mat<double>(n_por_capas.back(), output_n, arma::fill::randu) - 0.5)*0.6);
        bias.push_back(arma::Row<double>(output_n, arma::fill::zeros));
    }

    void print_red()
    {

        for (int i = 0; i < w.size(); ++i)
        {
            auto m_capa_actual = w[i];
            auto bias_actual = bias[i];

            cout << m_capa_actual << endl;
            cout << bias_actual;
        }
    }

    static activation_function sigmoid()
    {
        auto f = [](arma::Row<double> net) -> arma::Row<double>
        {
            // cout<<"aaa"<<endl;
            return 1.0 / (1.0 + arma::exp(-net));
        };
        auto df = [f](arma::Row<double> net) -> arma::Row<double>
        {
            return f(net) % (1.0 - f(net));
        };
        // activation_function sigmoid

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

            // cout<<(test >= arma::Row<double>(size(test),arma::fill::zeros) );
            // cout<<result;

            return arma::conv_to<arma::Row<double>>::from(result);
            // return {};
        };
        // activation_function sigmoid

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
        // activation_function sigmoid

        return activation_function{f, df, "soft_max"};
    }

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

    // arma::Row<double> activation(arma::Row<double> data){

    //     return activationSigmoid(data);

    // }

    double Error(arma::Row<double> &X, arma::Row<double> &Y)
    {
        return arma::sum(arma::square(pred(X) - Y) / 2);
    }

    void backpropagate(arma::Row<double> &X, arma::Row<double> &Y,int thrw)
    {

        arma::Row<double> actual = X;
        // vector<arma::Row<double>> sj_by_layers;
        vector<arma::Row<double>> net_by_layers;

        // cout<<"El tamanio de activation es: "<<activation.size()<<endl;
        // cout<<"El tamanio de w es: "<<w.size()<<endl;
        // cout<<activation.size()<<endl;
        for (int i = 0; i < w.size(); ++i)
        {
            arma::Row<double> net = actual * w[i] + bias[i];
            actual = activation[i].f(net);

            // cout<<i<<endl;
            net_by_layers.push_back(net);
            // sj_by_layers.push_back(actual);
        }

        // cout<<arma::size(X)<<" "<<arma::size(Y)<<endl;
        // cout<<sj_by_layers.back()<<endl;

        deque<arma::Mat<double>> w_derivates;
        deque<arma::Row<double>> bias_derivates;
        arma::Row<double> ro;

        for (int current_layer = net_by_layers.size() - 1; current_layer >= 0; --current_layer)
        {
            arma::Row<double> net = net_by_layers[current_layer];
            arma::Row<double> sj = activation[current_layer].f(net);
            //input data
            arma::Row<double> net_wj = current_layer == 0 ? X : activation[current_layer - 1].f(net_by_layers[current_layer - 1]);

            // cout<<"current_layer: "<<current_layer<<endl;

            // cout<<"sj: "<<arma::size(sj)<<endl;

            if (current_layer == net_by_layers.size() - 1)
            {
                //(dL/dsj)*(dsj/dNet) = dL/dNet

                // arma::Row<double> dsj_Netj = (sj) % (1 - sj);
                
                //F'(input data @ w + b)
                arma::Row<double> dsj_Netj = activation[current_layer].df(net);

                if (floss.name == "cross_entropy" && activation[current_layer].name == "soft_max")
                {
                    // cout<<"a" ;
                    ro = (sj - Y);
                }
                else
                    //dL/dsj
                    ro = floss.dloss(sj, Y) % dsj_Netj;
                
                if(thrw){
                    cout<<"net"<<endl;
                    cout<<net<<endl;
                    cout<<"dsj_netj"<<endl;
                    cout<<dsj_Netj<<endl;
                    cout<<"ro----"<<endl;
                    cout<<ro<<endl;
                    cout<<"sj----"<<endl;
                    cout<<sj<<endl;
                    cout<<"Y----"<<endl;
                    cout<<Y<<endl;
                    cout<<"dloss----"<<endl;
                    cout<<floss.dloss(sj,Y)<<endl;
                }


                // ro = (sj - Y) % dsj_Netj; //error

                // cout<<"ro: "<<size(ro)<<endl;
                //(dNet/dwj)*(dL/dNet)
                w_derivates.push_front(net_wj.t() * ro);
                // cout<<w_derivates.front().row(0);
                // cout<<w.back().row(0);
                // cout<<"ultima capa"<<endl;
                // throw;
                bias_derivates.push_front(ro);
                // cout<<size(w_derivates.front())<<endl;
            }
            else
            {
                arma::Mat<double> net_next_h_sj = w[current_layer + 1];

                //F'(net)
                arma::Row<double> dsj_Netj = activation[current_layer].df(net);
                arma::Row<double> ro_act = (net_next_h_sj * (ro.t())).t();
                ro = ro_act % (dsj_Netj);

                // auto ro_act = ro % dsj_Netj;
                // ro = ro_act * w[current_layer].t();


                w_derivates.push_front(net_wj.t() * ro);
                bias_derivates.push_front(ro);
            }

            // cout<<endl;
        }


        for (int i = 0; i < w.size(); ++i)
        {
            // cout<<"Derivadas capa "<<i<<endl;
            // cout<<w_derivates[i]<<endl;
            w[i] = w[i] - alpha * w_derivates[i];
            bias[i] = bias[i] - alpha * bias_derivates[i];
        }
    }

    pair<vector<double>, vector<double>> train(arma::Mat<double> &X_train, arma::Mat<double> &Y_train, int epoch, int print = 1, arma::Mat<double> X_validation = {}, arma::Mat<double> Y_validation = {})
    {
        vector<double> loss_training;
        vector<double> loss_validation;
        vector<int> indexes(size(X_train).n_rows);

        std::iota(indexes.begin(),indexes.end(),0);

        for (int i = 0; i < epoch; ++i)
        {
            random_shuffle(indexes.begin(),indexes.end());
            // cout<<i<<endl;
            double actual_error_train = 0;
            double actual_error_val = 0;
            for (int r = 0; r < indexes.size(); ++r)
            {

                int row = indexes[r];
                arma::Row<double> actual_X = X_train.row(row);
                arma::Row<double> actual_Y = Y_train.row(row);


                actual_error_train += floss.loss(pred(actual_X), actual_Y);
                backpropagate(actual_X, actual_Y,0);
            }

            for (int row = 0; row < size(X_validation).n_rows; ++row)
            {
                arma::Row<double> actual_X = X_validation.row(row);
                arma::Row<double> actual_Y = Y_validation.row(row);

                actual_error_val += floss.loss(pred(actual_X), actual_Y);
            }

            loss_training.push_back(actual_error_train);
            if(size(X_validation).n_rows)
                loss_validation.push_back(actual_error_val);

            if (print)
            {
                cout << "El error para la epoca " << i << " es " << actual_error_train << " " <<  actual_error_val << endl;
                // cout<<"El tu mama despues: "<<endl;
                // cout<< w[0] <<endl;
                // cout<<"si: "<<endl;
                // cout<< w.back()<<endl;
            }
        }
        return {loss_training, loss_validation};
    }



    arma::Row<double> multi_pred(arma::Mat<double> &X)
    {
        arma::Row<double> actual = X;

        return actual;
    }

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
            // cout<<"Leyendo :"<< wlist[i]<<" "<<blist[i]<<endl;
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

    // cout<<name<<endl;
    if(name == "") return {};
    fstream file;
    string buffer;
    arma::Mat<double> matrix;

    file.open(name, ios::in);

    // cout << file.is_open() << endl;

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