clear
clc
close all

%%%%%%%%%%%
%exercise2_sine
% A script for the solution of exercise 2, approximating the sine function.
%%%%%%%%%%%

%% Discussion hyperparameters!
noise = true;
use_early_stopping = false;
algorithms = ["traingd", "traingdx", "trainlm", "traincgf", "trainbfg", "trainbr"];
algorithms = ["trainlm"];
num_iterations_list = [200];
patience = 50; % early stopping
%hidden_layer_size_list = [5:10:100];
hidden_layer_size_list = 45;
num_repetitions = 1;

%% Generate data

% Generate training data
lower_bound = 0;
upper_bound = 3*pi;
dx = 0.01;

train_x = lower_bound : dx : upper_bound;
train_y = sin(train_x.^2);

n_test_points = 100;
test_x = lower_bound + (upper_bound - lower_bound)*rand(1, n_test_points);
test_y = sin(test_x.^2);

sigma=0.5;
train_yn = train_y + sigma*randn(size(train_y));
test_yn  = test_y + sigma*randn(size(test_y));

% Choose targets
if noise
    target = train_yn;
else
    target = train_y;
end

%% Train-validation split (if early stopping)
% Get train indices now
num_samples = size(target, 2);  % Total number of samples
all_indices = 1:num_samples;


% Prepare to get a validation set:
if use_early_stopping
    train_ratio = 0.8;               % 80% of data for training
    train_max_index = floor(num_samples * train_ratio);
    train_indices = all_indices(1:train_max_index);
    val_indices   = all_indices(train_max_index+1:end);
    
else
    % No noise: all for training
    train_indices = all_indices;
    val_indices   = [];
end

%% Define hyperparameters to be searched

% Create a new CSV file for saving the data
filename = 'sine_results.csv';
header = {'Hidden', 'Algorithm', 'Iterations', 'Training Time', 'Train MSE', 'Val MSE', 'Test MSE', 'Test MSE noise', 'Noise'};
writecell(header, filename);

for a = 1:length(num_iterations_list)
    num_iterations = num_iterations_list(a);
    for b = 1:length(hidden_layer_size_list)
        hidden_layer_size = hidden_layer_size_list(b);
        fprintf("Hidden layer size: %d\n", hidden_layer_size )
        for i = 1:length(algorithms)
            fprintf('Training algorithm %d out of %d\n', i, length(algorithms));
            alg = algorithms(i);
            for j = 1:num_repetitions
                fprintf('--- Training %d out of %d\n', j, num_repetitions);
                %% Set-up network
                net = feedforwardnet(hidden_layer_size, alg);
                % Configure to the example
                net = configure(net, train_x, target);
                % Set the train and validation indices for fixed split
                net.divideFcn = 'divideind';
                net.divideParam.trainInd = train_indices;
                net.divideParam.valInd = val_indices;
                net.divideParam.testInd = [];   % No test set
                % Randomly initialize the weights
                net = init(net);
                % Don't show window during training (annoying)
                net.trainParam.showWindow = 0;
                % Save the number of epochs to train
                net.trainParam.epochs = num_iterations;
                net.trainParam.max_fail = patience; % Patience for early stopping
                
                %% Training
                tic;
                [net, tr] = train(net, train_x, target);
                training_time = toc;
                % Evaluate the network on the testing data and training data
                y_pred = net(train_x);
                train_mse = mean((train_y - y_pred).^2);
                y_pred = net(test_x);
                test_mse   = mean((test_y  - y_pred).^2);
                test_mse_n = mean((test_yn - y_pred).^2);
                
                %% Save results
                results = {hidden_layer_size, char(algorithms(i)), num_iterations, training_time, tr.best_perf, tr.best_vperf, test_mse, test_mse_n, noise};
                writecell(results, filename, 'WriteMode', 'append');
            end
        end
    end
end

disp("Done!");

% Get visual performance for plots
x_vals = lower_bound : dx : upper_bound;
simulated = sim(net, x_vals);
writematrix(x_vals, "Data/xvals.csv");
writematrix(simulated, "Data/sine_simulated_noise.csv");