function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. (with regard to) to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

%X
%X(1, 3)

fprintf('size of theta: ');
size(theta)
theta
fprintf('\n');

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%				(y is a list)
%
% Note: grad should have the same dimensions as theta
%

% cost function
sum = 0; 		% init sum
for i = 1:m 	% sum together
	sum += ((-y(i) * log(1/(1+(1/e^(theta(1)+theta(2)*X(i, 2)+theta(3)*X(i, 3)))))) - ((1 - y(i)) * log(1 - 1/(1+(1/e^(theta(1)+theta(2)*X(i, 2)+theta(3)*X(i, 3)))))));
endfor
J = sum/m;		% get mean

% gradient

sum1 = 0;
sum2 = 0;
sum3 = 0;
for i = 1:m
	sum1 += ((1/(1+(1/e^(theta(1)+theta(2)*X(i, 2)+theta(3)*X(i, 3))))) - y(i)) * X(i, 1);
	sum2 += ((1/(1+(1/e^(theta(1)+theta(2)*X(i, 2)+theta(3)*X(i, 3))))) - y(i)) * X(i, 2);
	sum3 += ((1/(1+(1/e^(theta(1)+theta(2)*X(i, 2)+theta(3)*X(i, 3))))) - y(i)) * X(i, 3);
	endfor;
grad(1) = sum1/m;
grad(2) = sum2/m;
grad(3) = sum3/m;

% =============================================================

end
