clear
clc
close all

%%%%%%%%%%%
%exercise2_sine
% A script for the solution of exercise 2, approximating the sine function.
%%%%%%%%%%%

% Define the training algorithms to compare
algorithms = ["traingd", "trainlm", "traincgf", "trainbfg", "trainbr", ];

% Generate training data
lower_bound = 0;
upper_bound = 3*pi;
dx = 0.025;
train_x = lower_bound : dx : upper_bound;
train_y = sin(train_x.^2);

% Generate testing data
n_test_points = 50;
test_x = lower_bound + (upper_bound - lower_bound)*rand(1, n_test_points);
test_y = sin(test_x.^2);

% Also create perturbed (gaussian noise) training target data
sigma=0.2;
train_yn = train_y + sigma*randn(size(train_y));
% Choose our targets (noise or no noise)
noise = false;
if noise
    target = train_yn;
else
    target = train_y;
end

% Define the number of iterations, how often we train for an algorithm, and hidden layer size
num_iterations_list = [10, 100, 300];
num_repetitions = 20;
hidden_layer_size_list = [10, 50, 100];

% Create a new CSV file for saving the data
filename = 'sine_results.csv';
header = {'Hidden', 'Algorithm', 'Iterations', 'Training Time', 'Train MSE', 'Test MSE', 'Noise'};
writecell(header, filename);

%%% Do many iterations of training

% Vary over the amount of epochs for training
for a = 1:length(num_iterations_list)
    num_iterations = num_iterations_list(a);
    % Vary over the size of the hidden layer
    for b = 1:length(hidden_layer_size_list)
        hidden_layer_size = hidden_layer_size_list(b);
        for i = 1:length(algorithms)
            % Vary over the different training algorithms
            fprintf('Training algorithm %d out of %d\n', i, length(algorithms));
            for j = 1:num_repetitions
                % Repeat each set up 20 times
                fprintf('--- Training %d out of %d\n', j, num_repetitions);
                % Choose the algorithm to use
                alg = algorithms(i);
                % Define the network
                net = feedforwardnet(hidden_layer_size, alg);
                % Configure to the example
                net = configure(net, train_x, target);
                % Use only training data
                net.divideFcn = 'dividetrain';
                % Randomly initialize the weights
                net = init(net);
                % Don't show window during training (annoying)
                net.trainParam.showWindow = 0;
                % Save the number of epochs to train
                net.trainParam.epochs = num_iterations;
                % Start training, and time it
                tic;
                net = train(net, train_x, target);
                training_time = toc;
                % Evaluate the network on the testing data and training data
                y_pred = net(train_x);
                train_mse = mean((train_y - y_pred).^2);
                y_pred = net(test_x);
                test_mse = mean((test_y - y_pred).^2);
                
                % Save the training results to the CSV file
                results = {hidden_layer_size, char(algorithms(i)), num_iterations, training_time, train_mse, test_mse, noise};
                writecell(results, filename, 'WriteMode', 'append');
            end
        end
        disp("Done!");
    end
end

