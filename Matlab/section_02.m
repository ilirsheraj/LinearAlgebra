%% Matrix Types

% square vs rectangular
S = randn(5); %square matrix 5x5
S = randn(5,5); % square matrix 5x5
R = randn(5,2);

% identity
I = eye(3);

% Zeros
Z = zeros(4);

% Diagonal
D = diag([1 2 3 5 2]); % automatically fill up zeroes on off-diagonal elements

% Triangular matrix from full matrices
S = randn(5);
U = triu(S);
L = tril(S)

% Concatenate Matrices: Make sure numbe rof rows is the same
A = randn(3,2);
B = randn(3,4);
C = [A B]

%% Matrix addition and subtraction

A = randn(5,4);
B = randn(5,3);
C = randn(5,4);

% Try to add
% A + B; error
A + C

% Shift a matrix
l = 0.3;
N = 5;
D = randn(N);

Ds = D + l*eye(N);

[D, Ds]

%% Scalar Multiplication

M = [1 2; 2 5];
s = 2;

M*s
s*M

%% Code Challenge
% Test for some random MXN matrices whether s(A+B) = aA + sB
% Create two random matrices
A = randn(3,3);
B = randn(3,3);

% define a scalar
s = 2;

% test whether both sides of equation are equal
res1 = s*(A + B);
res2 = s*A + s*B;

res1 - res2

%% Matrix Transposition

M = [1 2 3; 2 3 4];

M'

M''

% Be careful of matrices with complex numbers
C = [4+1i 3 2-4i];
transpose(C)
C' % will flip the signs of the complex component (Hermitian)
C.' % will transpose as we want

%% Trace and Diagonal
M = round(5*randn(4));

% extract the diagonal
d = diag(M);

% notice two ways of using the diag function
d = diag(M);
D = diag(d); %diagonal matrix

% trace
tr = trace(M);

tr2 = sum(diag(M));

%% Code Challenge
% Is trace a linear operator?
% determine the relationship between tr(A) + tr(B) and tr(A+B)
% determine the relationship between tr(l*A) and l*tr(A)
% if these conditions true, then it is a linear operator

% geenrate two random matrices
A = round(4* randn(4, 4));
B = round(4*randn(4,4));

% calculate traces
tra = trace(A);
trb = trace(B);
trab = trace(A + B);
tra + trb == trab

% define a scalar
l = 5;

% trace of lambda with A
trace(l*A) == l*trace(A)
% yes true

%% Broadcasting Matrix Arithmetic
% create a matrix

A = reshape(1:12, 3, 4);

% create two vectors
r = [10 20 30 40];
c = [100 200 300]';

% Three methods of broadcasting

% The repmat way
A + repmat(r, size(A,1),1)
A + repmat(c, 1, size(A,2))

% the bsxfun way (binary expanssion)
bsxfun(@plus, A, r)
bsxfun(@plus, A, c)

% the non-mathy, newest way
A + r
A + c