function plot_weights(weights)
%% Function description
%
% The function takes in the weights of a neural network model as a cell array
% and creates visualizations of the weights and biases using four different methods: heatmap,
% scatter plot, network diagram, and histogram.
%
% The function first extracts the weight matrices and bias vectors for each layer of the neural
% network. It then combines all the weight matrices and bias vectors into two large matrices,
% one for all the weights and one for all the biases.
%
% The heatmap visualization shows a color-coded representation of the weights and biases, where
% each value in the matrix is represented by a different color.
%
% The scatter plot visualization shows the relationship between the weights and biases, with
% the weights plotted on the x-axis and biases plotted on the y-axis.
%
% The network diagram visualization shows a directed graph representation of the weights or
% biases, where each node represents a neuron and the edges represent the weights or biases
% connecting the neurons.
%
% The histogram visualization shows the distribution of the weights and biases across all
% layers using a histogram.

%% Extract weight matrices

W = cell(1,34);
b = cell(1,34);

for i = 1:34
    layer_start_idx = (i-1)*2+1;
    W{i} = weights{layer_start_idx};
    b{i} = weights{layer_start_idx+1};
end

% Combine weights and biases for all layers

all_W = cell2mat(W);
all_b = cell2mat(b);

%% Heatmap

figure;
imagesc(all_W);
title('All Weights Heatmap');
colorbar;

figure;
imagesc(all_b);
title('All Biases Heatmap');
colorbar;

%% Scatter plot

figure;
scatter(all_W(:), all_b(:), 'filled');
title('All Weights vs. Biases Scatter Plot');
xlabel('Weights');
ylabel('Biases');

%% Network diagram

figure;
g = digraph(all_W);
plot(g);
title('All Weights Network Diagram');

figure;
g = digraph(all_b);
plot(g);
title('All Biases Network Diagram');

%% Histogram

figure;
hold on;
histogram(all_W(:), 'FaceColor', 'blue', 'Normalization', 'probability');
histogram(all_b(:), 'FaceColor', 'red', 'Normalization', 'probability');
title('Histogram of All Weights and Biases');
legend('Weights', 'Biases');
hold off;
end
