clear;
clf;
close all;
clc;

%set(groot, 'defaultAxesFontSize', 16)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Time series prediction on logmap dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Load data and windowize
load Data/logmap.mat;

% Orders to check:
order_list = 2:1:100;
order_list = 89;

order_list=45;

%order_list=[4];

filename = 'Data/logmap_order_results.csv';
header = {'Order','Val MSE', 'Test MSE'};
writecell(header, filename);

for order=order_list

    X = windowize(Z, 1:(order + 1));
    Y = X(:, end);
    X = X(:, 1:order);
    
    %% Build model
    
    [gam, sig2, cost] = tunelssvm({X,Y,'f', [], [], 'RBF_kernel'}, 'simplex', 'crossvalidatelssvm', {10, 'mse'}) ;
    [alpha , b] = trainlssvm({X, Y, 'f', gam , sig2 });

    %% Do prediction:

    % TODO - wrong here? "order + 1", but visually, not true!
    Xs = Z(end - order + 1 : end, 1);
    nb = length(Ztest);
    prediction = predict({X, Y, 'f', gam , sig2 }, Xs , nb);
    test_mse = mean(mean((Ztest - prediction).^2));
    
    % Save results
    results = {order, cost, test_mse};
    writecell(results, filename, 'WriteMode', 'append');
end

disp("MSE");
disp(test_mse);


figure ;
hold on;
plot(Ztest(1:nb) , 'k');
plot(prediction , 'r');
hold off;

% Save data
writematrix(Z, "Data/logmap_train.txt");
writematrix(Ztest, "Data/logmap_test.txt");
writematrix(prediction, "Data/logmap_pred.txt");