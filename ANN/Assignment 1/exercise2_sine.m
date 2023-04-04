clear
clc
close all

%%%%%%%%%%%
%exercise2_sine
% A script for the solution of exercise 2, approximating the sine function.
%%%%%%%%%%%

% Define the training algorithms to compare
algorithms = ["traingd", "trainlm", "trainbfg"];

% Generate training data
dx = 0.05;
x = 0 : dx : 3*pi;
y = sin(x.^2);
% Add gaussian noise
sigma=0.2;
yn = y + sigma*randn(size(y));
% Choose our targets - change to yn for noisy data
t = yn;

% Define the number of iterations and hidden layer size
num_iterations = 100;
num_repetitions = 20;
hidden_layer_size = 50;

% Create a new CSV file for saving the data
filename = 'sine.csv';
header = {'Algorithm', 'Iterations', 'Training Time', 'Mean Squared Error'};
writecell(header, filename);

for i = 1:length(algorithms)
    fprintf('Training algorithm %d out of %d\n', i, length(algorithms));
    for j = 1:num_repetitions
        fprintf('--- Training %d out of %d\n', j, num_repetitions);
        % Choose the algorithm to use
        alg = algorithms(i);
        % Define the network
        net = feedforwardnet(hidden_layer_size, alg);
        % Configure to the example
        net = configure(net, x, t);
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
        net = train(net, x, y);
        training_time = toc;
        % Evaluate the network on the training data
        y_pred = net(x);
        mse = mean((y - y_pred).^2);
        
        % Save the training results to the CSV file
        results = {char(algorithms(i)), num_iterations, training_time, mse};
        writecell(results, filename, 'WriteMode', 'append');
    end
end
disp("Done!");