%% Matrix Inverse
% Generate a matrix of random numbers 3x3
m = 3;
A = randn(m);

% Compute the inverse
Ainv = inv(A);

% Check it is corrrect
id = A*Ainv;

% Show in an image
figure(1), clf
subplot(131), imagesc(A)
title("Matrix A"), axis square, axis off

subplot(132), imagesc(Ainv)
title("Matrix A^(-1)"), axis square, axis off

subplot(133), imagesc(id)
title("AA^(-1)"), axis square, axis off

%% Implement MCA in code
% Compute the inverse and compare it to the inv() function

m = 6;
A = randn(m);
rank(A)

% Part 1: Compute the minors matrix
rows = true(1,m);
rows(1) = false;

cols = true(1,m);
cols(1) = false;

A(rows, cols);

% Initialize
[minors,H] = deal(zeros(m));

% Create a loop
for i=1:m
    for j=1:m
        % define rows and columns for submatrix
        rows = true(1,m);
        rows(i) = false;

        cols = true(1,m);
        cols(j) = false;

        minors(i,j) = det(A(rows, cols));

        % Compute the H-Matrix
        H(i,j) = (-1)^(i+j);
    end
end

% the cofactor matrix
C = minors .*H;

% compute the adjugate
Ainv = C'/det(A);

Ainv - inv(A)

imagesc(A*Ainv)

%% Inverse via row reduction
m = 4;
A = round(10*randn(m));

% Augment A annd its identity
Aaug = [ A eye(m)];

% take reduced row eschelon
Asol = rref(Aaug);

Ainv = inv(A);

% Visualize
figure(2), clf

subplot(211), imagesc(Aaug)
title("A|I"), axis("off")
set(gca, "clim", [-1 1]*max(abs(Aaug(:)))*0.25)

subplot(212), imagesc(Asol)
title("I|A^{-1}"), axis("off")
set(gca, "clim", [-1 1]*max(abs(Asol(:)))*0.25)

figure(3), clf

subplot(311), imagesc(A)
title("Matrix A"), axis("off")
set(gca, "clim", [-1 1]*max(abs(A(:)))*0.25)

subplot(312), imagesc(Asol(:, 5:8))
title("A^{-1} from rref"), axis("off")
set(gca, "clim", [-1 1]*max(abs(Asol(:)))*0.08)

subplot(313), imagesc(Ainv)
title("A^{-1} from inv() function"), axis("off")
set(gca, "clim", [-1 1]*max(abs(Ainv(:)))*0.25)

%% Code Challenge: Inverse of diagonal Matrices

% Create some diagonal matrices: start with 2x2 and move up
% Compute their inverse (condition on diagonal matrix for invertibility)

A = [2 0; 0 3];
% need to install the tools first :) 
sym( inv(A) )

% Another way to create a diagonal matrix
A = diag(1:5);

sym(inv(A))

B = diag(1:10);

Adiag = diag(B);
Idiag = diag(inv(B));

Adiag .* Idiag;

% elements squared
Adiag ./ Idiag

% conditions for invertibility of diagonal matrix

%% Left- and Right-Inverse
% m>n for left inverse
% m<n for right inverse
m = 6;
n = 3;

A = randn(m,n);
AtA = A'*A;
AAt = A*A';

disp(["Rank of AtA is " num2str(rank(AtA)) ])
disp(["Rank of AAT is " num2str(rank(AAt)) ])

% Left inverse
Aleft = inv(AtA)*A';
I_left = Aleft * A;

% Right Inverse
Aright = A'*inv(AAt);
I_right = A * Aright; % because it is not full-rank

% Inverse function
AtA_inv = inv(AtA);
I_AtA = AtA_inv*AtA;

AAt_inv = inv(AAt);
I_AAt = AAt_inv*AAt;

% show images
figure(4), clf
subplot(331), imagesc(A), axis image, axis off
title([ 'A, r=' num2str(rank(A)) ])

subplot(332), imagesc(AtA), axis image, axis off
title([ 'A^TA, r=' num2str(rank(AtA)) ])

subplot(333), imagesc(AAt), axis image, axis off
title([ 'AA^T, r=' num2str(rank(AAt)) ])

subplot(335), imagesc(Aleft), axis image, axis off
title('Left inverse: (A^TA)^{-1}A^T')

subplot(336), imagesc(Aright), axis image, axis off
title('Right inverse: A^T(AA^T)^{-1}')

subplot(338), imagesc(I_left), axis image, axis off
title('[ (A^TA)^{-1}A^T ]  A')

subplot(339), imagesc(I_right), axis image, axis off
title('A  [ A^T(AA^T)^{-1} ]')

%% Pseudoinverse
% pseudoinverse of rectangular matrix A
pseudoInvA = pinv(A);

subplot(334), imagesc(pseudoInvA), axis image, axis off
title('MP Pseudoinverse of A')


subplot(337), imagesc(pseudoInvA*A), axis image, axis off
title('A^*A')

% create random matrix
n = 50;
A = randn(n);

% make rank deficient by repeating a column
A(:,end) = A(:,end-1);

% rank of A!
rank(A)

%% Pseudoinverse = inverse when matrix is invertible
% matrix size
m = 4;

% random integers matrix
A = round( 10*randn(m) );

% augment A and identity
Aaug = [ A eye(m) ];
size(Aaug)

% rref
Asol = rref(Aaug);

Ainvrref = Asol(:,m+1:end);
Ainv = inv(A);


% show the augemented matrices
figure(2), clf
subplot(211), imagesc(Aaug)