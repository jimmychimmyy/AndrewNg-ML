function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
% === reg cost === %
J = (1/(2*m)) * sum(((X*theta - y)).^2);
theta_2 = theta(2:end);							% remove first row from theta
J += (lambda/(2*m)) * sum(theta_2.^2); 

% === gradient === %
% === for j = 0 === %
for i = 1:m
	grad(1) += ((theta(1) + theta(2) * X(i, 2)) - y(i)) * X(i, 1);
end

% === for j >= 1 === %
for j = 2:rows(theta)
	for i = 1:m
		grad(j) += ((theta(1) + theta(2) * X(i, 2)) - y(i)) * X(i, j) + (lambda/m) * theta(j);
	end
end
grad = grad/m;

% =========================================================================

grad = grad(:);

end
