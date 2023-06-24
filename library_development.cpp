#include "Red.h"

//archivo usado para probar la librería

int main(int argc, char **argv)
{
    arma::arma_rng::set_seed(0);

    vector<int> capas = {5, 2};
    vector<activation_function> activation = {Red::sigmoid(), Red::sigmoid()};

    // Red rn(4, capas, //  activation ,Red::sigmoid(), 3, 0.1);

    Red rn(4, capas, 3, activation, Red::sigmoid(), Red::mse_loss(), 0.1);

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
    auto Y = (read_data("./datasets/iris_clases.csv"));
    // arma::Row<double> Y = (read_data("./datasets/iris_pred.csv")).t();

    // cout << arma::size(X) << endl;
    // cout << arma::size(Y) << endl;

    // rn.print_red();
    // cout << "\nVerificacion de tipos: " << (Red::soft_max() == Red::soft_max());

    rn.train(X, Y, 1000);


    // arma::Row<double> test = {1,1,1,12,31,-12315512,154,-1251351551,0,-12315,-015315};

    // cout<<(test >= arma::Row<double>(size(test),arma::fill::zeros) );
    // cout<<(Red::relu()).f(test);
    // cout<<(Red::relu()).df(arma::Row<double>{},test);

    // cout<<endl;
    // for (int i = 0; i < size(X).n_rows; ++i)
    // {
    //     arma::Row<double> r = X.row(i);
    //     arma::Row<double> r_real = Y.row(i);
    //     cout << rn.pred(r) << " " << r_real << endl;
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