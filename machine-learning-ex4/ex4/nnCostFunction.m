function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y, :);

% Part 1 : Cost without Regularization
A1 = [ones(1, m) ; X'];
Z2 = Theta1 * A1;
A2 = sigmoid(Z2);
Z3 = Theta2 * [ones(1, m) ; A2];
A3 = sigmoid(Z3);
Hx = A3';

J_matrix = (-1 .* y_matrix .* log(Hx)) + ((y_matrix - 1) .* log(1 - Hx));
J = (sum(sum(J_matrix, 1), 2))/m;

% Part 1 : Add regularization to cost

reg_term1 = sum(sum(Theta1(:, 2:size(Theta1, 2)).^2, 1), 2);
reg_term2 = sum(sum(Theta2(:, 2:size(Theta2, 2)).^2, 1), 2);
reg_term = ((reg_term1 + reg_term2) * lambda) / (2 * m);
J = J + reg_term;

% Part 2 : Backpropagation (Vectorized method)

% Step 1 : Feedforward
a1 = [ones(1, m) ; X'];
z2 = Theta1 * a1;
a2 = sigmoid(z2);
z3 = Theta2 * [ones(1, m) ; a2];
a3 = sigmoid(z3');
% Size now is m x num_classes_output_layer

% Step 2 : Output layer gradient
del3 = a3 - y_matrix;

% Step 3 : Hidden layer gradient
del2 = ((Theta2(:,2:end))' * del3')' .* sigmoidGradient(z2');
% I was probably getting the incorrectness because I was multiplying del3 to Theta2 here

% Step 4 : Calculate capital (triangular) deltas
Delta1 = del2' * a1';
Delta2 = del3' * [ones(1, m) ; a2]';
% I was probably getting the incorrectness because I was not initiating ones here

% Step 5 : Final Gradient computation
Theta1_grad = Delta1 ./ m;
Theta2_grad = Delta2 ./ m;

% Part 3 : Add regularization to backpropagation
Reg_terms_1 = Theta1(:, 2:end) .* (lambda/m);
Reg_terms_2 = Theta2(:, 2:end) .* (lambda/m);
Regd_terms_1 = Theta1_grad(:, 2:end) + Reg_terms_1;
Regd_terms_2 = Theta2_grad(:, 2:end) + Reg_terms_2;
Theta1_grad = [Theta1_grad(:,1) Regd_terms_1];
Theta2_grad = [Theta2_grad(:,1) Regd_terms_2];

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
