clear
clc
close all

%%%%%%%%%%%
%exercise2_personal
% A script for the solution of exercise 2, the personal regression problem.
%%%%%%%%%%%


% Load the datafile as a structure array
data = importdata('Data/Data_Problem1_regression.mat');

% Store the variables in the workspace
X1 = data.X1;
X2 = data.X2;
T1 = data.T1;
T2 = data.T2;
T3 = data.T3;
T4 = data.T4;
T5 = data.T5;

% Digits for the student number r0708518
d1 = 8;
d2 = 8;
d3 = 7;
d4 = 5;
d5 = 1;

% Build personal dataset
Tnew = (d1*T1 + d2*T2 + d3*T3 + d4*T4 + d5*T5)/(d1 + d2 + d3 + d4 + d5);
% Export the data so we can plot it in Python
my_data = [X1, X2, Tnew];
writematrix(my_data, "Data/my_data.csv");

% Sample: Training and validation data 
n_points = 2000;
n = length(X1);
idx = randperm(n, n_points);

train_X1   = X1(idx);
train_X2   = X2(idx);
train_Tnew = Tnew(idx);

train_input  = [train_X1'; train_X2'];
train_target = train_Tnew';

% Sample: test data 
n_points = 1000;
idx = randperm(n, n_points);
test_X1   = X1(idx);
test_X2   = X2(idx);
test_Tnew = Tnew(idx);

test_input  = [test_X1'; test_X2'];
test_target = test_Tnew';

%%% Train neural network

% Sizes of the hidden layers to be explored:
num_layers_list = [3];
sizes_list = [10];
% Max number iterations during training
num_iterations = 1000;
% Train with LM algorithm
alg = "trainbr";
% To save results to external file
filename = "personal_regression.csv";

% To initialize the file:
header = {'Num layers', 'Num hidden','Train MSE', 'Val MSE', 'Test MSE'};
writecell(header, filename);

num_repetitions = 1;

for num_layer=num_layers_list
    for sizee=sizes_list
        hidden_layer_size = repelem(sizee, num_layer);
        disp("---");
        disp("Hidden layer size: ");
        disp(hidden_layer_size);
        for a=1:num_repetitions
            fprintf("Rep: %d/%d | ", a, num_repetitions)
            net = feedforwardnet(hidden_layer_size, alg);
            % Save the number of epochs to train
            net.trainParam.epochs = num_iterations;
            % Split into validation and training set for early stopping
            net.divideFcn = 'dividerand';
            net.divideParam.trainRatio = 0.7;
            net.divideParam.valRatio   = 0.3;
            net.divideParam.testRatio  = 0;
            % Save the number of validation checks before early stopping 
            net.trainParam.max_fail = 10;
            
            % Set hidden layer activation functions to tansig
            num_hidden_layers = numel(net.layers) - 1;
            for i = 1:num_hidden_layers
                net.layers{i}.transferFcn = 'tansig';
            end
            %%% Choose whether we show training or not
            net.trainParam.showWindow = 0;
            % Train the network
            [net, tr] = train(net, train_input, train_target);
            % See training results: plotperform(tr)
            
            % Get performance on the test set
            predictions = net(test_input);
            % Get differences as well
            differences = (test_target - predictions).^2;
            test_perf = mean(differences);
            results = {num_layer, sizee, tr.best_perf, tr.best_vperf, test_perf};
            writecell(results, filename, 'WriteMode', 'append');
        end
    end
end