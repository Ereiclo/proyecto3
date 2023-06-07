#include <iostream>
#include <armadillo>
#include <deque>
#include <vector>

using namespace std;




class Red{

    vector<arma::Mat<double>> capas;
    double alpha = 0.15;

    public:

    Red(int input_n, vector<int>&capas,int output_n,double alpha): alpha{alpha}{

        // this->capas[0](3,3);
        this->capas.push_back(arma::Mat<double>(input_n,capas[0],arma::fill::randu));


        for(int i = 1; i < capas.size();++i){
            int n = capas[i-1];
            int m = capas[i];
            this->capas.push_back(arma::Mat<double>(n,m,arma::fill::randu));
        }   

        this->capas.push_back(arma::Mat<double>(capas.back(),output_n,arma::fill::randu));

    }

    void print_red(){


        for(auto& m: this->capas){
            cout<<m<<endl;
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

    void backpropagate(arma::Row<double>& X, arma::Row<double> &Y){

        arma::Row<double> actual = X;
        vector<arma::Row<double>> sj_by_layers;


        for(int i = 0; i < capas.size();++i){
            actual = activation(actual*this->capas[i]);
            sj_by_layers.push_back(actual);
        }


        deque<arma::Mat<double>> derivates;

        for(int current_layer = sj_by_layers.size()-1;;--current_layer){
            arma::Row<double> sj = sj_by_layers[current_layer];
            arma::Row<double> ro;
            arma::Row<double> net_wj = current_layer == 0 ? X : sj_by_layers[current_layer-1];



            if(current_layer == sj_by_layers.size()-1){
                //(dL/dsj)*(dsj/dNet) = dL/dNet

                arma::Row<double> dsj_Netj = (sj)%(1 - sj);
                ro = (sj - Y)% dsj_Netj;
                                    //(dNet/dwj)*(dL/dNet)
                derivates.push_front(net_wj.t()*ro);
            }
            else{
                arma::Mat<double> net_next_h_sj = capas[current_layer+1];
                        //dL/sj*(dj/dNetj)    
                arma::Row<double> dsj_Netj = (sj)%(1 - sj);
                ro = (net_next_h_sj*(ro.t())*(dsj_Netj.t())).t();

                derivates.push_front(net_wj.t()*ro);

            }

        }

        for(int i = 0; i < capas.size();++i){
            capas[i] = capas[i] - alpha*derivates[i];
        }

    }


    arma::Row<double> pred(arma::Row<double>& X){
        arma::Row<double> actual = X;


        for(int i = 0; i < capas.size();++i){
            actual = activation(actual*this->capas[i]);

        }


        return actual;

    }





};


int main(int argc, char** argv){
    // arma::arma_rng::set_seed_random();
    vector<int> capas = {10,2};

    Red rn(1,capas,1);

    // rn.print_red();


    // arma::Row<double> input(1);
    arma::Col<double> test({2,3,4,5});
    arma::Col<double> test2({3,7,13,6});


    // cout<<rn.activationRelu(test);

    cout<<  test % test2;

    // auto s = arma::size(nada);





    // input(0) = 2;

    // cout<<input;


    // cout<< rn.pred(input);



    return 0;
}