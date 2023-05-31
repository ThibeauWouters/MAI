clear;
clf;
close all;
clc;

%set(groot, 'defaultAxesFontSize', 16)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Time series prediction on logmap dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Load data and windowize
load Data/santafe.mat;

% Normalize the dataset

mu = mean(Z);
s = std(Z);

Z = (Z - mu)/s;
Ztest = (Ztest - mu)/s;

% Performance measure
measure = 'mse';

% Orders to check:
order_list = 2:1:100;
%order_list=[50];
order_list = 100;

filename = 'Data/santafe_order_results.csv';
header = {'Order','Val MSE', 'My MSE', 'Test MSE'};
writecell(header, filename);

% For tuning, get my own validation set
my_validation_set_start = 544;
my_validation_set = Z(my_validation_set_start+1:my_validation_set_start + 99);

for order=order_list

    X = windowize(Z, 1:(order + 1));
    Y = X(:, end);
    X = X(:, 1:order);
    
    %% Train model
    
    [gam, sig2, cost] = tunelssvm({X,Y,'f', [], [], 'RBF_kernel'}, 'simplex', 'crossvalidatelssvm', {10, measure}) ;
    [alpha , b] = trainlssvm({X, Y, 'f', gam , sig2 });

    %% MSE

    % My validation set
    
    Xs = Z(my_validation_set_start - order + 1 : my_validation_set_start);
    nb = length(my_validation_set);
    prediction = predict({X, Y, 'f', gam , sig2 }, Xs , nb);
    my_val_mse = mean((my_validation_set - prediction).^2);
    % Real test set:
    Xs = Z(end - order + 1 : end, 1);
    nb = length(Ztest);
    prediction = predict({X, Y, 'f', gam , sig2 }, Xs , nb);
    test_mse = mean(mean((Ztest - prediction).^2));
    
    % Save results
    results = {order, cost, my_val_mse, test_mse};
    writecell(results, filename, 'WriteMode', 'append');
end

%% Plot
figure ;
hold on;
plot(Ztest(1:nb) , 'k');
plot(prediction , 'r');
hold off;

figure;

disp("MSE");
disp(test_mse);

% Save data
writematrix(Z, "Data/santafe_train.txt");
writematrix(Ztest, "Data/santafe_test.txt");
writematrix(prediction, "Data/santafe_pred.txt");