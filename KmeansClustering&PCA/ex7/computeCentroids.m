function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

concat = horzcat(idx, X);
sort(concat);

for i = 1:K
	ind = concat(:, 1) == i;
	A = concat(ind, :);
	centroids(i, 1) = sum(A(:, 2))/rows(A);
	centroids(i, 2) = sum(A(:, 3))/rows(A);
end

%{
ind1 = concat(:, 1) == 1;
ind2 = concat(:, 1) == 2;
ind3 = concat(:, 1) == 3;

A1 = concat(ind1, :)
A2 = concat(ind2, :)
A3 = concat(ind3, :)

centroids(3, 1) = sum(A3(:, 2))/rows(A3)
centroids(3, 2) = sum(A3(:, 3))/rows(A3)

%}

% =============================================================


end

