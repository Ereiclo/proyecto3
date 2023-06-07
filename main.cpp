#include <iostream>
#include <armadillo>

using namespace std;




class Red{

    vector<arma::Mat<double>> capas;

    public:

    Red(int input_n, vector<int>&capas,int output_n){

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

        return activationRelu(data);

    }


    double Error(arma::Row<double>& X,  arma::Row<double> &Y) {
        return arma::sum(arma::square(pred(X) - Y)/2);
    }

    void backpropagate(arma::Row<double>& X, arma::Row<double> &Y){

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
    arma::Row<double> test({2,3,-1,-235,-3,-5});


    cout<<rn.activationRelu(test);


    // auto s = arma::size(nada);





    // input(0) = 2;

    // cout<<input;


    // cout<< rn.pred(input);



    return 0;
}