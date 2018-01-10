function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

if (isvector(z))
	for i = 1:size(z)
		g(i) = 1/(1+e^-(z(i)));
	endfor
elseif (ismatrix(z))
	col = columns(z); 	% get num of columns for z
	row = rows(z);		% get num of rows for z
	for i = 1:row
		for j = 1:col
			g(i, j) = 1/(1+e^(-z(i)))
		endfor
	endfor
else
	g = 1/(1+e^(-z));
endif
break;

% =============================================================

end
