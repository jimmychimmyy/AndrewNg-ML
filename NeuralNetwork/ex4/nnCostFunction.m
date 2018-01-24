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

%size(Theta1)		% 25x401
%size(Theta2)		% 10x26
%size(X)			% 5000x400
%size(y)			% 5000x1
%num_labels			% 10
%m 					% 5000

% === get hypothesis === %
bias = ones(size(X, 1), 1);				% create bias vector 
a1 = horzcat(bias, X);					% add bias to X (= a1) -> 5000x401
z2 = (a1*Theta1');
a2 = sigmoid(z2);
%a2 = sigmoid(X*Theta1');				% a2 is 5000x401 * 401x25 = 5000x25

bias2 = ones(size(a2, 1), 1);			% create bias vector
a2 = horzcat(bias2, a2);				% add bias to a2 -> 5000x26
z3 = Theta2*a2';
h = sigmoid(z3);
%h = sigmoid(Theta2*a2');				% 10x26 * 26x5000 = 10x5000


%size(h)			% 10x5000
% should i take the transpose of x?
% then get the max of each row?
% and create 10 dimensional vectors using the position of the max
%	 of each row as 1 and the other positions as 0?

% === calculate cost === %
result = zeros(size(h));		% 10x5000

for i = 1:m
	result(y(i), i) = 1;
end

total = 0;
for j = 1:m
	total += -result(:, j)' * log(h(:, j)) - (1 - result(:, j)') * log(1 - h(:, j));
end
J += (total/m);

% === calcualte regularization === %
theta1 = Theta1(:, 2:end);
theta2 = Theta2(:, 2:end);

total = 0;
for i = 1:size(theta1, 1)
	for j = 1:size(theta1', 1)
		total += theta1(i, j)^2;
	end
end

for i = 1:size(theta2, 1)
	for j = 1:size(theta2', 1)
		total += theta2(i, j)^2;
	end
end

J += (lambda/(2*m)) * total;


% === backpropagation === %
for t = 1:m

	% === perform feedforward === %

	% add bias term to layer a1
	a_1 = X(m, :)';
	a_1 = [1;a_1];
	%size(a_1)		%401x1

	z_2 = Theta1*a_1;
	a_2 = sigmoid(z_2);
	%size(a_2)		% 25x1

	% add bias term to layer a2
	a_2 = [1; a_2];
	%size(a_2)		%26x1

	z_3 = Theta2*a_2;
	a_3 = sigmoid(z_3);
	%size(a_3)		%10x1

	% === calculate delta at output layer === %

	delta_3 = zeros(size(a_3));
	%size(delta_3)		% 10x1

	for k = 1:size(a_3(:, 1))
		%a_3(k, :)
		%y(t) == k
		%a_3(k) - (y(t) == k)
		delta_3(k) = (a_3(k) - (y(t) == k));
	end

	z_2 = [1; z_2];		% not sure this is correct, but its the only way for the line below to run
	delta_2 = (Theta2' * delta_3) .* sigmoidGradient(z_2);

	%size(delta_2)		%26x1
	delta_2 = delta_2(2:end);		% remove bias

	delta_matrix_1 = zeros(size(delta_2*a_1'));
	delta_matrix_2 = zeros(size(delta_3*a_2'));

	delta_matrix_1 += delta_2*a_1';
	delta_matrix_2 += delta_3*a_2';

end


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
