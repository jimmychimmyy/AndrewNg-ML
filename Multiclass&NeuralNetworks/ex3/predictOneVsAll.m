function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);		% all_theta = 10, num_labels = 10

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

% return a vector of preditions for each ex in the matrix X
% X contains the ex in rows, X is a 5000x401 matrix
% p is 401x1 vector
% all_theta is 10x401
% we want 401x something matrix so either X'*all_theta 401x5000 * 10*401
% 

h = sigmoid(X*all_theta'); % h is size 5000 x 10

% we want to get the index of the column that p(i) is coming from

p = max(h, [], 2);

for i = 1:m
	row = h(i, :);
	for j = 1:num_labels
		if (p(i) == h(i, j))
			p(i) = j;
		end
	end
end

% =========================================================================


end
