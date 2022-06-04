%% Small and Large Singular Matrices

% generate a 2x2 matrix of integers, and with linear dependencies
% compute the rank
A = [1 3; 1 3];
det(A)
% generate mxm matric, impose linear dependencies
% compute rank
% for small and large m
m = 2;
A = randn(m);
rank(A)
det(A)

A(:,1) = A(:,2)
det(A) % its zero but because of comouter rounding its not exactly zero

m = 4
A = randn(m);
rank(A)
det(A)

A(:,1) = A(:,2)
det(A)
rank(A)

% Let's make a really large matrix
m = 15
A = randn(m);
rank(A)
det(A)

A(:,1) = A(:,2)
det(A)
rank(A)

% with very large matrices, determinant increases despite the theory of
% being zero
m = 30
A = randn(m);
rank(A)
det(A)

A(:,1) = A(:,2)
det(A)
rank(A)

m = 300
A = randn(m);
rank(A)
det(A)

A(:,1) = A(:,2)
det(A)
rank(A)

%% Code Challenge: Determinan of swaped rows
% Generate a 6x6 matrix and compute its determinant
A = randn(6);
disp(["Before row swap: " num2str(det(A))])

% swap its rows and compute again
As = A([2 1 3 4 5 6], :);
disp(["After row swap: " num2str(det(As))])

% swap two rows and compute again
Ass = A([2 1 3 5 4 6], :);
disp(["After row 2 swaps: " num2str(det(Ass))])

% Lets try the same for columns
As = A(:, [2 1 3 4 5 6]);
disp(["After column swap: " num2str(det(As))])

Ass = A(:, [2 1 3 5 4 6]);
disp(["After column 2 swaps: " num2str(det(Ass))])

%% Determinant of Shifted Matrices
% Generate a random square matrix (n=20)
% impose linear dependence
% Shift the matrix (0 -> .1 times the identity matrix) (lambda)
% compute the abs(determinant) of the shifted matrix
% repeat it 1000 times, take the average abs
% plot the determinant as a function of lambda

lambdas = linspace(0, 0.1, 30);
% initialize the determinant
dets = zeros(length(lambdas),1);
for deti = 1:length(lambdas)
    % run 1000 iterations
    for i=1:1000
        % generate a matrix
        M = randn(20);

        % reduce the rank
        M(:,1) = M(:,2);

        % Compute the determinant
        temp(i) = abs( det(M + lambdas(deti)*eye(20)));
    end
    dets(deti) = mean(temp);
end

plot(lambdas, dets, "s-")
xlabel("Fraction of identity matrix for shifting")
ylabel("Determinant")


%% Code Challenge: Determinant of Matrix Product
% Illustrate that det(AB) = det(A)*det(B)
% 1) For a random 3x3 matrix
% 2) In a loop over rabdom matrix sizes of up to 40x40
% 3) Plot their differences

A = randn(3);
B = randn(3);

deta = det(A);
detb = det(B);

detab = det(A*B);
disp([detab, deta*detb])

dets = zeros(50,1);
for i=1:50
    A = randn(i);
    B = randn(i);

    dets(i) = det(A*B) - det(A)*det(B);
end

plot(dets, "s-")
set(gca, "ylim", [-1,1])
% Good after 15, and then it blows up because of instability.


