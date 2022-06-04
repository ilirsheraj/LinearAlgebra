%% Matrix Rank
% Define matrix dimensions
m = 4;
n = 6;

% create a random matrix
A = randn(m,n);

% compute the rank
ra = rank(A);
disp([ "rank(A) = " num2str(ra)])

% set the last column to be the repeat of penultimate column
B = A;
B(:,end) = B(:, end -1);

rb = rank(B);
disp(["rank(B) = " num2str(rb)])

% change to rows
B = A;
B(end, :) = B(end - 1, :)

rb2 = rank(B);
disp(["rank(B) = " num2str(rb2)]) % now it fell to 3

%% Adding noise to a rank-deficient matrix

% create a square matrix for convenience
A = round(10* randn(m,m));
rank(A);
% reduce the rank
A(:,1) = A(:,2);
rank(A);

% introduce some noise
noiseamp = 0.001;

B = A + noiseamp*randn(size(A));

disp(" ")
disp(["rank w/o noise = " num2str(rank(A)) ])
disp(["rank with noise = " num2str(rank(B)) ]) % increases the rank

%% Code Challenge: Reduced Rank matrix via multiplication
% We want to create a method to create a matrix of any size that has almost
% any rank we want wrt to upper boundaries we saw

% create a 10x10 matrix with rank 4

% use matrix multiplication
A = randn(10,4) * randn(4,10);
size(A);
rank(A);

% generalize the procedure to create any mxn matrix with rank r
m = 8;
n = 47;
r = 3;

A = randn(10,r) * randn(r,n);
rank(A)

%% Code Challenge: Scalar Multiplication and Rank

% Create two matrices, a full- and a reduced-rank
m = 5;
n = 4;
A = randn(5,4);
B = A;
B(:, end) = B(:, end -1);

% another way to create a reduced matrix
R = randn(m, n-1);
rank(R);

% calculate their ranks
disp(["rank(A) = " num2str(rank(A)) ])
disp(["rank(B) = " num2str(rank(B)) ])

% define a scalar
l = 2

% display the ranks of matrices after scalar multiplication
disp(["rank(A) = " num2str(rank(l*A)) ])
disp(["rank(B) = " num2str(rank(l*B)) ])

% is the rank(l*A) equalt to l*rank(A)?
% yes it it
disp(["rank(A) = " num2str(l*rank(l*A)) ])
disp(["rank(B) = " num2str(l*rank(l*B)) ])

%% Rank of A and A^TA
% matrix sizes
m = 14;
n = 3;

% create matrices
A = round(10*randn(m,n));

AtA = A'*A;
AAt = A*A';

fprintf("\n AtA: %gx%g, rank = %g", size(AtA,1), size(AtA,2), rank(AtA))
fprintf("\n AAt: %gx%g, rank = %g\n", size(AAt,1), size(AAt,2), rank(AAt))

%% Code Challenge: Rank of Multiplied and Summed Matrix
% generate two matrices (rectangular)
m = 2;
n = 5;
A = randn(m,n);
B = randn(m,n);
rankA = rank(A);
rankB = rank(B);
% compute ATA and BTB
AtA = A'*A;
BtB = B'*B;

% Find their ranks
rank(AtA);
rank(BtB);

% calculate rank(ATA) and rank(BTB)
rank(AtA * BtB) 
% 2 is the maximum possible rank
rank(AtA + BtB) % less than or equalt to rank(AtA) + rank(BtB)

%% Full-Rank Matrix by shifting
m = 30;

A = randn(m);
A = round(10*A'*A);
rank(A)
imagesc(A);

% Reduce the rank
A(:,1) = A(:, 2);
rank(A) % 29

% create a lambda
l = 0.01;

% new matrix
B = A + l*eye(m);

disp(["rank without shift = ", num2str(rank(A))])
disp(["rank with shift = ", num2str(rank(B))])

%% Code Challenge: Vector in the span of a set
% determine whether this vector
v = [1 2 3 4]';

% is in the span of these sets
S = [ [4 3 6 2]' [0 4 0 1]' ];
T = [ [1 2 2 2]' [0 0 1 2]' ];

% Calculate the ranks of sets
rank(S)
rank(T)

% create augmented matrix
Sc = [S v];
rank(Sc)
Tc = [T v];
rank(Tc)

%% End!