clear
clc
close all

%%%%%%%%%%%
% exercise1_hop2d.m
% A script which generates n random initial points 
%and visualises results of simulation of a 2d Hopfield network 'net'
%%%%%%%%%%


savedata = true;
if savedata
    % Create a new, empty CSV file for saving the data
    filename = 'hop_2D_trajectories.csv';
    % Clear the contents of the CSV file
    fclose(fopen(filename, 'w'));
end

% Define the patterns we want to store
T = [1 1; -1 -1; 1 -1]';
net = newhop(T);

% Define the x and y ranges for the grid of points
x_range = linspace(-1, 1, 11);
y_range = linspace(-1, 1, 11);

% Create the mesh of grid points using meshgrid
[x, y] = meshgrid(x_range, y_range);

tic;
% iterate over the grid points
for i = 1:numel(x)
    x_val = x(i);
    y_val = y(i);
    % Define starting point
    start = {[x_val; y_val]};
    [path, Pf, Af] = sim(net, {1 50}, {}, start);
    record = [cell2mat(start) cell2mat(path)];
    if savedata
        writematrix(record, filename, 'WriteMode', 'append');
    end
end
timed = toc;
disp("Time:");
disp(timed);