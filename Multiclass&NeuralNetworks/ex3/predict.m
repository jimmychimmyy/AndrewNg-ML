function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


size(Theta1)					% 25 x 401
size(Theta2)					% 10 x 26
size(X) 						% 5000 x 400
size(p) 						% 5000 x 1
m 								% 5000
num_labels 						% 10


%[row, column] = size(X);
a1_ones = ones(size(X, 1), 1);

X = horzcat(a1_ones, X);

size(a1_ones)

%X(row, column+1) = 1;			% add column of ones to the matrix X = 5000 x 401
z2 = X*Theta1';					% (5000x401)(401x25) = 5000 x 25
a2 = sigmoid(z2);

%size(a2) 						% 5000 x 25

%[row, column] = size(a2);
a2_ones = ones(size(a2, 1), 1);

a2 = horzcat(a2_ones, a2);

%a2(row, column+1) = 1;			% add column of ones to the matrix a2 = 5000 x 26
z3 = a2*Theta2';				% (5000 x 26)(26 x 10) = 5000 x 10
h = sigmoid(z3);

size(h) 						% 5000 x 10


%h(5000, :)
%max(h(5000, :), [], 2)

p = max(h, [], 2);

for i = 1:m
	row = h(i, :);
	for j = 1:num_labels
		if (p(i) == h(i, j))
			p(i) = j;
		end
	end
end

% input layer is X
% hidden layer is sigmoid(theta1 * input layer (i))
% output layer is sigmoid(theta2 * hidden layer (i)) == hypothesis

% hypothesis will contain what we are looking for


% =========================================================================


end
