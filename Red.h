#include <iostream>
#include <fstream>
#include <armadillo>
#include <deque>
#include <vector>
#include <functional>

using namespace std;

struct activation_function
{
    function<arma::Row<double>(arma::Row<double>)> f;
    function<arma::Row<double>(arma::Row<double>, arma::Row<double>)> df;
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
        w.push_back(arma::Mat<double>(input_n, n_por_capas[0], arma::fill::randu));
        bias.push_back(arma::Row<double>(n_por_capas[0], arma::fill::zeros));

        for (int i = 1; i < n_por_capas.size(); ++i)
        {
            int n = n_por_capas[i - 1];
            int m = n_por_capas[i];
            w.push_back(arma::Mat<double>(n, m, arma::fill::randu));
            bias.push_back(arma::Row<double>(m, arma::fill::zeros));
        }

        w.push_back(arma::Mat<double>(n_por_capas.back(), output_n, arma::fill::randu));
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
        auto f = [](arma::Row<double> X) -> arma::Row<double>
        {
            // cout<<"aaa"<<endl;
            return 1 / (1 + arma::exp(-X));
        };
        auto df = [](arma::Row<double> Y, arma::Row<double> X) -> arma::Row<double>
        {
            return Y % (1 - Y);
        };
        // activation_function sigmoid

        return activation_function{f, df, "sigmoid"};
    }

    static activation_function tanh()
    {
        auto f = [](arma::Row<double> X) -> arma::Row<double>
        {
            return 2 / (1 + exp(-2 * X)) - 1;
        };
        auto df = [](arma::Row<double> Y, arma::Row<double> X) -> arma::Row<double>
        {
            return 1 - arma::square(Y);
        };

        return activation_function{f, df, "tanh"};
    }

    static activation_function relu()
    {
        auto f = [](arma::Row<double> X) -> arma::Row<double>
        {
            return arma::max(X, arma::Row<double>(size(X), arma::fill::zeros));
        };
        auto df = [](arma::Row<double> Y, arma::Row<double> X) -> arma::Row<double>
        {
            arma::Row<arma::uword> result = (X >= arma::Row<double>(size(X), arma::fill::zeros));

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
        auto f = [](arma::Row<double> X) -> arma::Row<double>
        {
            return arma::exp(X) / (arma::sum(arma::exp(X)));
        };
        auto df = [](arma::Row<double> Y, arma::Row<double> X) -> arma::Row<double>
        {
            return Y % (1 - Y);
        };
        // activation_function sigmoid

        return activation_function{f, df, "soft_max"};
    }

    static loss_function mse_loss()
    {

        auto loss = [](arma::Row<double> Y_pred, arma::Row<double> Y) -> double
        {
            return arma::sum(arma::square(Y_pred - Y) / 2);
        };

        auto dloss = [](arma::Row<double> Y_pred, arma::Row<double> Y) -> arma::Row<double>
        {
            return Y_pred - Y;
        };

        return loss_function{loss, dloss, "mse"};
    }

    static loss_function cross_entropy_loss()
    {

        auto loss = [](arma::Row<double> Y_pred, arma::Row<double> Y) -> double
        {
            return (-1) * arma::sum(Y % arma::log2(Y_pred));
        };

        auto dloss = [](arma::Row<double> Y_pred, arma::Row<double> Y) -> arma::Row<double>
        {
            double eps = 0.0000000000000001;

            return (-1) * (Y / (Y_pred + eps));
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

    void backpropagate(arma::Row<double> &X, arma::Row<double> &Y)
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
            arma::Row<double> net_wj = current_layer == 0 ? X : activation[current_layer - 1].f(net_by_layers[current_layer - 1]);

            // cout<<"current_layer: "<<current_layer<<endl;

            // cout<<"sj: "<<arma::size(sj)<<endl;

            if (current_layer == net_by_layers.size() - 1)
            {
                //(dL/dsj)*(dsj/dNet) = dL/dNet

                // arma::Row<double> dsj_Netj = (sj) % (1 - sj);
                arma::Row<double> dsj_Netj = activation[current_layer].df(sj, net);

                if (floss.name == "cross_entropy" && activation[current_layer].name == "soft_max")
                {
                    // cout<<"a" ;
                    ro = (sj - Y);
                }
                else
                    ro = floss.dloss(sj, Y) % dsj_Netj;

                // ro = (sj - Y) % dsj_Netj; //error

                // cout<<"ro: "<<size(ro)<<endl;
                //(dNet/dwj)*(dL/dNet)
                w_derivates.push_front(net_wj.t() * ro);
                bias_derivates.push_front(ro);
                // cout<<size(w_derivates.front())<<endl;
            }
            else
            {
                arma::Mat<double> net_next_h_sj = w[current_layer + 1];
                // dL/sj*(dj/dNetj)
                // cout<<"dNet(h+1)/dsj: "<<size(net_next_h_sj)<<endl;
                // cout<<"ro(h+1) = (dL/dNet(H+1)): "<< size(ro.t())<<endl;
                // arma::Row<double> dsj_Netj = (sj) % (1 - sj);
                arma::Row<double> dsj_Netj = activation[current_layer].df(sj, net);
                // cout<<"dsj/dNetj: "<< size(dsj_Netj.t())<<endl;
                ro = ((net_next_h_sj * (ro.t())).t() % (dsj_Netj));
                // cout<<"ro: "<<size(ro)<<endl;

                w_derivates.push_front(net_wj.t() * ro);
                bias_derivates.push_front(ro);
                // cout<<size(w_derivates.front())<<endl;
            }

            // cout<<endl;
        }

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

        for (int i = 0; i < epoch; ++i)
        {
            double actual_error_train = 0;
            double actual_error_val = 0;
            for (int row = 0; row < size(X_train).n_rows; ++row)
            {
                arma::Row<double> actual_X = X_train.row(row);
                arma::Row<double> actual_Y = Y_train.row(row);

                backpropagate(actual_X, actual_Y);
                actual_error_train += floss.loss(pred(actual_X), actual_Y);
            }

            for (int row = 0; row < size(X_validation).n_rows; ++row)
            {
                arma::Row<double> actual_X = X_validation.row(row);
                arma::Row<double> actual_Y = Y_validation.row(row);

                actual_error_val += floss.loss(pred(actual_X), actual_Y);
            }

            loss_training.push_back(actual_error_train);
            loss_validation.push_back(actual_error_val);

            if (i % 100 == 0 && print)
            {
                cout << "El error para la epoca " << i << " es " << actual_error_train << endl;
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

    return matrix;
}