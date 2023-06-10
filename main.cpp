#include <iostream>
#include <fstream>
#include <armadillo>
#include <deque>
#include <vector>

using namespace std;




class Red{

    vector<arma::Mat<double>> w;
    vector<arma::Row<double>> bias;
    // vector<arma::Mat<double>> f;
    double alpha = 0.15;

    public:

    Red(int input_n, vector<int>&n_por_capas,int output_n,double alpha): alpha{alpha}{

        // this->capas[0](3,3);
        w.push_back(arma::Mat<double>(input_n,n_por_capas[0],arma::fill::randu));
        bias.push_back(arma::Row<double>(n_por_capas[0],arma::fill::zeros));


        for(int i = 1; i < n_por_capas.size();++i){
            int n = n_por_capas[i-1];
            int m = n_por_capas[i];
            w.push_back(arma::Mat<double>(n,m,arma::fill::randu));
            bias.push_back(arma::Row<double>(m,arma::fill::zeros));
        }   

        w.push_back(arma::Mat<double>(n_por_capas.back(),output_n,arma::fill::randu));
        bias.push_back(arma::Row<double>(output_n,arma::fill::zeros));

    }

    void print_red(){


        for(int i = 0; i < w.size();++i){
            auto m_capa_actual = w[i];
            auto bias_actual = bias[i];

            cout<<m_capa_actual<<endl;
            cout<<bias_actual;
        }


    }

    arma::Row<double> activationSigmoid(arma::Row<double>data){

        return 1/(1 + exp(-data));
    }

    arma::Row<double> activationRelu(arma::Row<double> data){
        arma::Row<double> zeros(arma::size(data));

        return arma::max(data,zeros);
    }

    arma::Row<double> activation(arma::Row<double> data){

        return activationSigmoid(data);

    }


    double Error(arma::Row<double>& X,  arma::Row<double> &Y) {
        return arma::sum(arma::square(pred(X) - Y)/2);
    }

    void backpropagate(arma::Row<double> &X, arma::Row<double> &Y){

        arma::Row<double> actual = X;
        vector<arma::Row<double>> sj_by_layers;


        for(int i = 0; i < w.size();++i){
            actual = activation(actual*w[i] + bias[i]);
            sj_by_layers.push_back(actual);
        }

        // cout<<arma::size(X)<<" "<<arma::size(Y)<<endl;
        // cout<<sj_by_layers.back()<<endl;


        deque<arma::Mat<double>> w_derivates;
        deque<arma::Row<double>> bias_derivates;
        arma::Row<double> ro;

        for(int current_layer = sj_by_layers.size()-1; current_layer >= 0   ;--current_layer){
            arma::Row<double> sj = sj_by_layers[current_layer];
            arma::Row<double> net_wj = current_layer == 0 ? X : sj_by_layers[current_layer-1];

            // cout<<"current_layer: "<<current_layer<<endl;

            // cout<<"sj: "<<arma::size(sj)<<endl;



            if(current_layer == sj_by_layers.size()-1){
                //(dL/dsj)*(dsj/dNet) = dL/dNet

                arma::Row<double> dsj_Netj = (sj)%(1 - sj);
                ro = (sj - Y)% dsj_Netj;
                // cout<<"ro: "<<size(ro)<<endl;
                                    //(dNet/dwj)*(dL/dNet)
                w_derivates.push_front(net_wj.t()*ro);
                bias_derivates.push_front(ro);
                // cout<<size(w_derivates.front())<<endl;
            }
            else{
                arma::Mat<double> net_next_h_sj = w[current_layer+1];
                        //dL/sj*(dj/dNetj)    
                // cout<<"dNet(h+1)/dsj: "<<size(net_next_h_sj)<<endl;
                // cout<<"ro(h+1) = (dL/dNet(H+1)): "<< size(ro.t())<<endl;
                arma::Row<double> dsj_Netj = (sj)%(1 - sj);
                // cout<<"dsj/dNetj: "<< size(dsj_Netj.t())<<endl;
                ro = ( (net_next_h_sj*(ro.t())).t() % (dsj_Netj));
                // cout<<"ro: "<<size(ro)<<endl;

                w_derivates.push_front(net_wj.t()*ro);
                bias_derivates.push_front(ro);
                // cout<<size(w_derivates.front())<<endl;

            }


            // cout<<endl;

        }

        for(int i = 0; i < w.size();++i){
            w[i] = w[i] - alpha*w_derivates[i];
            bias[i] = bias[i] - alpha*bias_derivates[i];
        }

    }

    void train(arma::Mat<double> &X_train, arma::Mat<double> &Y_train, int epoch ){

        for(int i = 0; i < epoch;++i){
            double actual_error = 0; 
            for(int row = 0; row < size(X_train).n_rows;++row){
                arma::Row<double> actual_X = X_train.row(row);
                arma::Row<double> actual_Y = Y_train.row(row);

                backpropagate(actual_X,actual_Y);
                actual_error += Error(actual_X,actual_Y);
            }

            if( i % 100 == 0){
                cout<<"El error para la epoca "<<i<<" es "<<actual_error<<endl;
            }
        }


    }


    arma::Row<double> multi_pred(arma::Mat<double>& X){
        arma::Row<double> actual = X;

        


        return actual;

    }

    arma::Row<double> pred(arma::Row<double>& X){
        arma::Row<double> actual = X;


        for(int i = 0; i < w.size();++i){
            actual = activation(actual*w[i] + bias[i]);

        }


        return actual;

    }





};


vector<double> readLine(string line){

    stringstream ss(line);
    string item;
    vector<double> result;

    //separar la string line por ','
    while(getline(ss,item,',')){
        item.erase(remove(item.begin(),item.end(),' '),item.end());
        result.push_back(stod(item));
    }

    return result;
}


arma::Mat<double> read_data(string name){
    fstream file;
    string buffer;
    arma::Mat<double> matrix;

    file.open(name,ios::in);


    cout<<file.is_open()<<endl;
    
    
    getline(file,buffer,'\n');

    while(getline(file,buffer,'\n')){

        arma::Row<double> actual_data(readLine(buffer));


        if(matrix.is_empty()){
            matrix =  actual_data;
        }else{
            matrix = arma::join_cols(matrix, actual_data);
        }


    }


    return matrix;


}


int main(int argc, char** argv){
    arma::arma_rng::set_seed(0);

    vector<int> capas = {5,2};

    Red rn(4,capas,3,0.1);

    // rn.print_red();


    // arma::Row<double> input(1);
    // arma::Col<double> test({2,3,4,5});
    // arma::Col<double> test2({3,7,13,6});


    // cout<<rn.activationRelu(test);

    // cout<<  test % test2;

    // arma::Mat<double> test;

    // vector<double> v1 = {1,2,34};
    // vector<double> v2 = {1,30,34};

    auto X = (read_data("./datasets/iris.csv"));
    auto Y = (read_data("./datasets/iris_pred.csv"));
    // arma::Row<double> Y = (read_data("./datasets/iris_pred.csv")).t();

    cout<<arma::size(X)<<endl;
    cout<<arma::size(Y)<<endl;

    // rn.print_red();

    rn.train(X,Y,5000);



    // cout<<endl;
    // for(int i= 0; i < size(X).n_rows;++i){
    //     arma::Row<double> r = X.row(i);
    //     arma::Row<double> r_real = Y.row(i);
    //     cout<<rn.pred(r)<<" "<<r_real<<endl;
    // }

    // auto row2 = arma::conv_to<arma::rowvec>::from(v1);
    // auto row3 = arma::conv_to<arma::rowvec>::from(v2);

    // test = arma::join_cols(row2,row3);
    // test = row2;





    // cout<<arma::Mat<double>{}.is_empty();


    // cout<<test(2)<<endl;


    // auto s = arma::size(nada);





    // input(0) = 2;

    // cout<<input;


    // cout<< rn.pred(input);



    return 0;
}