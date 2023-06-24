#include "Red.h"

activation_function decide_activation(string name)
{
    if (name == "sigmoid")
        return Red::sigmoid();
    else if (name == "relu")
        return Red::relu();
    else if (name == "tanh")
        return Red::tanh();
    else if (name == "soft_max")
        return Red::soft_max();
    return activation_function{};
}

loss_function decide_loss(string name)
{
    if (name == "mse")
        return Red::mse_loss();
    else if (name == "cross_entropy")
        return Red::cross_entropy_loss();

    return loss_function{};
}

int main(int argc, char **argv)
{
    
    arma::arma_rng::set_seed(42);
    int input_size;
    vector<int> capas;
    vector<string> blist;
    vector<string> wlist;
    int capa_final;
    vector<activation_function> activation;
    activation_function activation_final;
    string base_dir;
    string training_X_name, training_Y_name;
    string test_X_name, test_Y_name;
    loss_function ls;
    double alpha;
    int loading = 0;
    int print = 1;
    int epoch;

    for (int i = 1; i < argc; ++i)
    {
        string actual_param = argv[i];
        stringstream ss(actual_param);
        string param_name;
        string value;

        getline(ss, param_name, ':');
        getline(ss, value);

        if (param_name == "input_size" || param_name == "capas" 
            || param_name == "capa_final" || param_name == "epoch" 
            || param_name == "alpha" || param_name == "print" || param_name == "loading")
        {
            vector<double> numbers = readLineNumbers(value);
            if (param_name == "input_size")
                input_size = numbers[0];
            else if (param_name == "capas")
                capas = vector<int>(numbers.begin(), numbers.end());
            else if (param_name == "capa_final")
                capa_final = numbers[0];
            else if (param_name == "alpha")
            {
                alpha = numbers[0];
            }
            else if (param_name == "epoch")
                epoch = numbers[0];
            else if (param_name == "print")
                print = numbers[0];
            else if(param_name == "loading"){
                loading = numbers[0];
                // cout<<"a"<<endl;
            }
        }
        else
        {
            vector<string> strings = readLineStrings(value);

            if (param_name == "activation")
                for (auto &act : strings)
                {
                    activation.push_back(decide_activation(act));
                }
            else if (param_name == "activation_final")
                activation_final = decide_activation(strings[0]);
            else if (param_name == "loss")
                ls = decide_loss(strings[0]);
            else if (param_name == "training_data")
            {
                training_X_name = strings[0];
                training_Y_name = strings[1];
            }
            else if (param_name == "validation_data")
            {
                test_X_name = strings[0];
                test_Y_name = strings[1];
            }else if(param_name == "base_dir"){
                base_dir = strings[0];
            }else if(param_name == "blist"){

                blist = strings;

            }else if(param_name == "wlist"){

                wlist = strings;

            }
        }

        // cout<<param_name<<" "<<value<<endl;
    }
    auto X_train = (read_data(training_X_name));
    auto Y_train = (read_data(training_Y_name));

    auto X_test = (read_data(test_X_name));
    auto Y_test = (read_data(test_Y_name));

    if (print)
    {

        cout << "El input size es " << input_size << endl;

        cout << "El valor para cada capa es: ";
        for (auto &c_value : capas)
        {
            cout << c_value << " ";
        }
        cout<<endl;

        cout << "El valor para la capa final es: " << capa_final << endl;

        cout << "El valor de la activation para cada capa es: ";
        for (auto &f_activation : activation)
        {
            cout << f_activation.name << " ";
        }
        cout<<endl;

        cout << "El valor para la activation final es: " << activation_final.name << endl;

        cout << "El valor de la loss function es: " << ls.name << endl;

        cout << "El nombre del archivo de train X es " << training_X_name << endl;
        cout << "El nombre del archivo de train Y es " << training_Y_name << endl;

        cout << "El nombre del archivo de test X es " << test_X_name << endl;
        cout << "El nombre del archivo de test Y es " << test_Y_name << endl;

        cout << "Epoch: " << epoch << " alpha: " << alpha << endl;

        cout << "Train data size: " << arma::size(X_train) << "\nTrain class size: "
             << arma::size(Y_train) << endl;

        cout << "Test data size: " << arma::size(X_test) << "\nTest class size: "
             << arma::size(Y_test) << endl;
        
        cout << "wlist: "<<endl;

        for(int i = 0; i < wlist.size();++i){
            cout<<wlist[i]<<" ";
        }
        cout<<endl;       


        cout << "blist: "<<endl;

        for(int i = 0; i < blist.size();++i){
            cout<<blist[i]<<" ";
        }
        cout<<endl;
    }

    Red rn(input_size, capas, capa_final, activation, activation_final, ls, alpha);

    pair<vector<double>, vector<double>> result; 


    
    if(!loading)
        result = rn.train(X_train, Y_train, epoch, print, X_test, Y_test);
    else 
        rn.load_red(wlist,blist);
    

    if(print){
        rn.show_red();
    }
    


    if(!loading)
        for(int i = 0; i < result.first.size();++i) {
            auto error = result.first[i];
            cout<<error<<(i == (result.first.size() -1 ) ? "" : " ");
        }
    cout<<endl;

    if(!loading)
        for(int i = 0; i < result.second.size();++i) {
            auto error = result.second[i];

            cout<<error<<(i == (result.second.size() -1 ) ? "" : " ");
        }
    cout<<endl;


    for(int i = 0; i < size(X_test).n_rows; ++i){
        arma::Row<double> r = X_test.row(i);
        
        cout<<arma::index_max(rn.pred(r))<<(i == (size(X_test).n_rows - 1) ? "" : " ");
        // cout<<(rn.pred(r))<<endl;
    }

    if(base_dir != ""){
        rn.save_red(base_dir);
    }

    return 0;
}
