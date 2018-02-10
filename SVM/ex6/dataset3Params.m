function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% uncomment following block to compute optimum C and sigma
%{ 
test_C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
test_sigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

prediction = svmPredict(svmTrain(Xval, yval, C, @(Xval, yval) gaussianKernel(Xval, yval, sigma)), Xval);
min_err = mean(double(prediction ~= yval))

% predictions is vetor containing all predictions from SVM
% need to try with different values of C and sigma
% we are minimizing error, therefore we want the smallest
for i = 1:numel(test_C)
	for j = 1:numel(test_sigma)
		%test_C(i)
		%test_sigma(j)
		predictions = svmPredict(svmTrain(Xval, yval, test_C(i), @(Xval, yval) gaussianKernel(Xval, yval, test_sigma(j))), Xval);
		err = mean(double(predictions ~= yval))
		if (err < min_err)
			C = test_C(i);
			sigma = test_sigma(j);
			min_err = err;
		end
	end
end

min_err
C
sigma

%}

% found that smallest error is when C = 10 and sigma = 0.03
C = 10;
sigma = 0.03;


% =========================================================================

end
