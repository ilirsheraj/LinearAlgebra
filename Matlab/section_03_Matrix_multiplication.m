%% Intro to Matrix Multuplication

m = 4;
n = 3;
k = 6;

% create some matrices
A = randn(m,n);
B = randn(n,k);
C = randn(m,k);

% test multiplication validity
A*B; % YES
A*A; % NO
A'*C; % YES
B*B; % NO
B'*B; % YES
B*C; % NO
C*B; % NO
C'*B; % NO
C*B'; % YES

%% Layer-Wise Matrix Multiplication
%Generate two matrices
m = 4;
n = 6
A = randn(m,n);
B = randn(n,m);

% Build matrix product layerwise
C1 = zeros(m);
for i = 1:n
    C1 = C1 + A(:,i) * B(i,:);
end

% implement matrix multiplication directly
C2 = A*B;
% compare
C1 - C2

%% Order of Operations
n = 12;
L = randn(n);
I = randn(n);
V = randn(n);
E = randn(n);

% forward multiplication and then transpose
res1 = (L*I*V*E)';

% Result of flipped multiplication and transpose each matrix
res2 = E'*V'*I'*L';

% Test their differences

res1 - res2

% they are the same

%% Matrix-Vector Multiplication
m = 4;

N = round(10*randn(m));

% scaled Symmetric Matrix
S = round(N'*N/m^2); 

% vector
w = [-1 0 1 2]';

% Multiplications with symmetric matrix
S*w
(S*w)'
w'*S'
w'*S

% Multiplications with nonsymmetric matrices
N*w
(N*w)'
%w*N
w'*N'
w'*N

%% 2D Transformation Matrices
% create a 2D input vector
v = [3 -2];

% create a square matrix
A = [1 -1;
    2 -1];

% Multiply them
w = A*v';

% plot the whole thing
figure(1), clf

plot([0 v(1)], [0 v(2)], "k", "linew", 2)
hold on
plot([0 w(1)], [0 w(2)], "r", "linew", 2)

% make the plot a bit better
axis square
axis([-1 1 -1 1]*max([norm(v) norm(w)]))
hold on
plot(get(gca, "xlim"), [0 0], "k--")
plot([0 0 ], get(gca, "ylim"), "k--")
legend({"v"; "Av"})
title("Rotation & Streching")

% define a rotation matrix
v = [3 -2];
% specify the rotation angle in radians
theta = pi/30;

% 2x2 transformation matrix (pure rotation)
% A = [cos(theta) -sin(theta);
%     sin(theta) cos(theta)];
% 2x2 transformation matrix sreeching
A = [2*cos(theta) -sin(theta);
    sin(theta) 2*cos(theta)];

% output vector
w = A*v';

% plot the whole thing
figure(2), clf

plot([0 v(1)], [0 v(2)], "k", "linew", 2)
hold on
plot([0 w(1)], [0 w(2)], "r", "linew", 2)

% make the plot a bit better
axis square
axis([-1 1 -1 1]*max([norm(v) norm(w)]))
hold on
plot(get(gca, "xlim"), [0 0], "k--")
plot([0 0 ], get(gca, "ylim"), "k--")
legend({"v"; "Av"})
title("Rotation & Streching")

%% Code challenge
% use a for loop to see the relationship between theta and Av magnitude
v = [3 -2]';
thetas = linspace(0, 2*pi,100);
vecmags = zeros(length(thetas), 2);

for i = 1:length(thetas)
    % rotation angle
    theta = thetas(i);
    % 2x2 impure transformation matrix
    A1 = [2*cos(theta) -sin(theta);
        sin(theta) 2*cos(theta)];

    % 2x2 pure transformation matrix
    A2 = [cos(theta) -sin(theta);
        sin(theta) cos(theta)];

    % output: vector magnitude
    vecmags(i,1) = norm(A1*v);
    vecmags(i,2) = norm(A2*v);
end

clf
plot(thetas, vecmags, "linew", 3)
ylabel("Av Magnitude"), xlabel("Angle in Radians")
legend({"Impure Rotation"; "Pure Rotation"})

%% Code Challenge 2: Geometric Transformations
% generate xy coordinates for a circle and plot it
x = linspace(-pi, pi, 100);
xy = [cos(x); sin(x)]';

figure(4), clf
plot(xy(:,1), xy(:,2), "bs", "MarkerFaceColor","w")
% create a 2x2 matrix (starting with I)
% change one of the coordinates
T = [1 0; 2 1];

% multiply the matrix by coordinates
newxy = xy*T

% plot the new coordinates
hold on
plot(newxy(:,1), newxy(:,2), "rs", "MarkerFaceColor","m")
axis([-1 1 -1 1]*max(abs([newxy(:); xy(:)])))
axis square

% play with various matrices

% try a singular matrix as well (to be seen later)
% T = [1 2; 2 4]
% it flattens out the circle into a single dimensional line because of the
% linearly dependent set

%% Additive and Multiplicative Identity Matrices

% matrix size
n = 4;
A = round(10*randn(n));
I = eye(n);
Z = zeros(n);

% test each case
isequal(A*I, A)
isequal(A, A*I)
isequal(A, A+I)

isequal(A, A + Z)
isequal(A, A*Z)
isequal(A+I, A+Z)

%% Additive and multiplicative symmetric matrices
m = 5;
n = 5;

% create two matrices: Additive
A = randn(m,n);
S = (A+A')/2

% Multiplicative
m = 5;
n = 3;

A = randn(m, n);
AtA = A'*A;
AAt = A*A';

% Show that they are square
size(AtA)
size(AAt)

% Show that they are symmetric
AtA - AtA'
AAt - AAt'

%% Hadamard Multiplication

m = 13;
n = 2;

A = randn(m,n);
B = randn(m,n);

C = A.*B

%% Code challenge: COmbine Symmetric Matrices
% Create two symmetric matrices

m = 5;
A = round(3*randn(m,m));
B = round(4* randn(m,m));

A_s = A'*A;
B_s = B'*B;

% Sum of symmetric matrices
Ma = A_s + B_s; % Symmetric

% Matrix Multiplication
Mm = A_s*B_s; % Nor Symmetric

% Hadamard Multiplication
Mh = A_s.*B_s; % Symmetric

% Test: SYmmetric: Matrix = Matrix transpose

Ma - Ma' % yes

Mm - Mm' % No

Mh - Mh' % yes

%% Multiplication of two Symmetric Matrices

% syms a b c d e f g h k l m n o p q r s t u

% Symmetric and constant-diagonal matrices

% A = [a b c d;
%      b a e f;
%      c e a h;
%      d f h a];
% 
% B = [l m n o;
%      m l q r;
%      n q l t;
%      o q r l];
% 
% % COnfirm
% A - A.'
% B - B.'
% 
% diag(A)
% diag(B)
% 
% A*B - (A*B).'

% It is not a symmetric matrix

% lets create a submatrix
% n = 2;
% A1 = A(1:n, 1:n);
% B1 = B(1:n, 1:n);
% 
% A1*B1 - (A1*B1).';

% The rule works for 2x2 matrices but not higher.

%% Diagonal Matrix Multiplication

% Create two matrices (4x4): "full" and diagonal
A = randn(4);
D = diag(randn(4,1));

% Multiply each matrix by itself: Standadrd and Hadamard
A*A % standard
A.*A % Hadamard

D*D
D.*D % Give the same result

%% Fourier Transform

n = 52;
F = zeros(n,n);
w = exp(-2*pi*sqrt(-1)/n);
for j = 1:n
    for k = 1:n    
        m = (j-1)*(k-1);
        F(j,k) = w^m;
    end
end

imagesc(real(F));
imagesc(imag(F));
imagesc(abs(F));

x = randn(n,1);
X1 = F*x;
X2 = fft(x);

clc
plot(1:n, abs(X1));
hold on
plot(1:n, abs(X2), "o");

clf
plot(F(:,3))
plot(real(F(:,3)))

%% Forbenius Dot Product

% pick matrices of any size
m = 9;
n = 4;

A = randn(m, n);
B = randn(m, n);

% vectorize and then take vector dot product
Av = A(:);
Bv = B(:);

frob_dp = sum(Av.*Bv);

% trace method
frob_dp2 = trace(A'*B);

% Matrix Norm
Anorm = norm(A, "fro"); % fro = frobenius
Anorm2 = sqrt(trace(A'*A));
% they are both the same

%% Matrix Norms

% Create a matrix
A = [1 2 3; 4 5 6; 7 7 9];

% Frobenius norm
normFrob = norm(A, "fro");

% induced 2-norm

normInd2 = norm(A);

% Schatten p-norm
p = 1;
s = svd(A);
normSchat = sum(s.^p)^(1/p);

% All norms together
disp([normFrob normInd2 normSchat])

[Q, R] = qr(randn(5));

%% Conditions for Self-Adjoint
m = 5; % size
A = randn(m); % matrix
A = A*A'; % Symmetric Matrix

v = randn(5,1); % vector v
w = randn(5,1); % vector w

dot(A*v, w) - dot(v, A*w) % dot product differences

%% Code Challenge: Symmetry Index

%%% Part1: Matrix Assymetry INdex
Aanti = (A - A')/2; % Assymetric part of the matrix
MAI = norm(Aanti) / norm(A); % ratio of norms

%%% Compute MAI for symmetric, skew-symmetric and random matrix
A = randn(5);
A = (A + A')/2;
MAI = norm((A - A')/2)/ norm(A); % Zer, as expected

% create a skewed symmetric matrix
A = randn(5);
A = (A - A')/2 % skewed symmetric

MAI = norm((A-A')/2)/ norm(A); % 1, as expected

% Random Matrix
A = randn(5);
MAI = norm((A-A')/2)/norm(A) % something in between

%%% Formula for mixing skew/symmetric matrices
A = round(10*randn(5));
p = 1; % test with p = 1 (skewed) and p = 0 (perfectly symmetric)
A = (1-p)*(A+A') + p*(A-A')

%%% Test on random matrics
ps = linspace(0, 1, 100);
mai = zeros(size(ps));
for i = 1:length(ps)
    p = ps(i);
    A = randn(5);
    A = (1-p)*(A+A')/2 + p*(A-A')/2;
    mai(i) = norm((A - A')/2)/ norm(A);
end

plot(ps, mai, "s-")
% The end!