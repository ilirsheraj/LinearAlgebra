%% Singular Value Decomposition

% Define a matrix
A = [3 0 5; 8 1 3];

[U,S,V] = svd(A);

figure(1), clf
subplot(141), imagesc(A)
axis square, axis off, title("A")

subplot(142), imagesc(U)
axis square, axis off, title("U")

subplot(143), imagesc(S)
axis square, axis off, title("\Sigma")

subplot(144), imagesc(V')
axis square, axis off, title("V^T")

[A*V(:,1) S(1)*U(:,1)]

%% Code Challenge: SVD vs. eigendecomposition for square symmetric matrices
% Create a symmetric marix, eg, 5x5
% compute eigandecomposition (W,L) and SVD (U,S,V)
% Create images of all matrices
% Compare U and V, and U and W

A = randn(5);
A = A'*A;

%eig and sort
[W,L] = eig(A);
[d,sidx] = sort(diag(L), "descend");
% recreate L based on sorted vlaues
L = diag(d);
W = W(:,sidx); %rows preserved, columns changed


%svd
[U,S,V] = svd(A);

figure(2), clf
subplot(231), imagesc(W), title("W (eig)"), axis square
subplot(232), imagesc(L), title("\Lambda (eig)"), axis square
subplot(234), imagesc(U), title("U (svd)"), axis square
subplot(235), imagesc(S), title("S (svd)"), axis square
subplot(236), imagesc(V), title("V (svd)"), axis square

%% Relation between Singular Values and Eiganvalues
% Case 1: eig(A'*A) vs svd(A)

A = [3 1 0; 1 1 0];

[sort(nonzeros(eig(A'*A))) sort(svd(A)).^2] % They are the same

% Case 2: eig(A'*A) vs svd(A'*A)

[sort(eig(A'*A)) sort(svd(A'*A))]

% Case 3a: eig(A) vs svd(A), real-valued eigs
A = [3 1 0; 1 1 0; 1 1 1];

[sort(eig(A)) sort(svd(A))] % not the same, and no clear relation between them

% case 3b: random number, likely to give complex eiganvalues
A = randn(3);
[sort(eig(A)) sort(svd(A))] % again no relation between them
% svd are real values in random matrices

%% Code Challenge: U from Eigandecomposition of A
% Create a 3x6 matrix

A = randn(3,6);

% full svd (Us,Ss,Vs)
[Us, Ss, Vs] = svd(A);
Vs'*Vs;

% eig(A'*A) (V,L)
[V,L] = eig(A'*A);

% confirm that V = Vs
[d,sidx] = sort(diag(L), "descend");
L = diag(d);
V = V(:,sidx);

[V Vs]; % matrix A has rank of 3, so three columns will not be the same
% This has to do with the null space. They are equal in row space

% Check the reationship between Ss and L (take the square of Ss)
[diag(Ss).^2 diag(L(1:3,1:3))]

% Create U using only A,V and L
U = zeros(3);

for i=1:3
    U(:,i) = A*V(:,i)/sqrt(L(i,i))
end
U;

% Confirm thar U = Us
[U Us]
U - Us
U + Us

%% Code Challenge: A^TA, Av, and singular vectors

m = 14;
A = randn(m);

AtA = A'*A;
AAt = A*A';

[U,S,V] = svd(AtA);

diffs = zeros(m,1);
for i=1:m
    diffs(i) = sum((AAt*A*U(:,i) - A*U(:,i)*S(i,i)).^2);
end

%% Spectral Theory of Matrices

m = 40;
n = 30;

% define a 2d Gaussian for Smoothing
k = round((m+n)/4);
[X,Y] = meshgrid(linspace(-3,3,k));
g2d = exp(-(X.^2 +Y.^2)/(k/8));
imagesc(g2d);

% matrix
A = conv2(randn(m,n), g2d,"same");

% svd
[U,S,V] = svd(A);

Ascaled = A*1;

% SVD (need only singular values)
s = svd(Ascaled);

% color limit based on matrix range
clim = [-.5 .5]*max(abs(A(:)));

% show the constituent matrices
figure(2), clf
subplot(241), imagesc(A)
axis square, axis off, title('A')
set(gca,'clim',clim)

subplot(242), imagesc(U)
axis square, axis off, title('U')

subplot(243), imagesc(S)
axis square, axis off, title('\Sigma')
set(gca,'clim',clim)

subplot(244), imagesc(V')
axis square, axis off, title('V^T')

subplot(212)
plot(diag(S),'ks-','linew',2,'markersize',10,'markerfacecolor','w')
xlabel('Component number'), ylabel('\sigma')
title('"Scree plot" of singular values')

% Most of the information in this matrix can be accounted for by 5 values

%% now show the first 5 "layers" separately

figure(3), clf

rank1mats = zeros(5,m,n);

for i=1:5
    
    % create rank1 matrix
    rank1mats(i,:,:) = U(:,i)*S(i,i)*V(:,i)';
    
    subplot(2,5,i)
    imagesc(squeeze(rank1mats(i,:,:)))
    axis square, axis off
    set(gca,'clim',clim)
    title([ 'Component ' num2str(i) ])
    
    subplot(2,5,i+5)
    imagesc(squeeze(sum(rank1mats(1:i,:,:),1)))
    axis square, axis off
    set(gca,'clim',clim)
    title([ 'Components 1:' num2str(i) ])
end

%% SVD for low-rank approximation
% run tge code from previous section
nComps = 5;

% reduced vectors
Ur = U(:, 1:nComps);
Sr = S(1:nComps, 1:nComps);
Vr = V(:,1:nComps);

% low rank approximation
reconImage = Ur*Sr*Vr';

rank(reconImage)

% error map and percent difference from the original matrix
errormap = (reconImage - A).^2;
pctdiff = 100*norm(reconImage - A)/norm(A);

figure(4),clf
subplot(131), imagesc(A)
axis square, axis off, set(gca, "clim", clim), title("A")

subplot(132), imagesc(reconImage)
axis square, axis off, set(gca, "clim", clim), title("Recon A")

subplot(133), imagesc(errormap)
axis square, axis off, set(gca, "clim", clim), title("Error Map")

% put SVD results into structure (ease of size comparison)
mapRecn.U = Ur;
mapRecn.S = Sr;
mapRecn.V = Vr;

% get variable information
reconsize = whos('mapRecn');
origsize  = whos('A');

% display compression
disp([ num2str(100-100*reconsize.bytes/origsize.bytes) '% compression and ' ...
       num2str(pctdiff) '% error for r=' num2str(nComps) ' low-rank approximation.' ])

%% Convert Singular Values to Percent Variance

% define matrix size
m = 40;
n = 30;

% define a 2D Gaussian for smoothing
k = round((m+n)/4);
[X,Y] = meshgrid(linspace(-3,3,k));
g2d = exp(-(X.^2 + Y.^2)/(k/8));

% Define the matrix
A = conv2(randn(m,n),g2d,"same");

Ascaled = A*14535654;

% SVD (need only singular values)
s = svd(Ascaled);

% convert to percent variance
spct = 100*s/sum(s);

% plot the singular values for comparison
% This will change according to scalar multiplication
figure(6), clf
subplot(211)
plot(s,'ks-','linew',2,'markersize',10,'markerfacecolor','w')
xlabel('Component number'), ylabel('\sigma')
title('Raw singular values')

% this will not change no matter the scalar used to multiply the matrix
subplot(212)
plot(spct,'ks-','linew',2,'markersize',10,'markerfacecolor','w')
xlabel('Component number'), ylabel('\sigma (% of total)')
title('Percent-change-normalized singular values')

%% Code Challenge: When is UV^T Valid
% Generate a matrix s.t U*VT is valid (only for square matrix)
A = randn(5);
[U,S,V] = svd(A);

% compute the norm of U, V and U*VT
norm(U) %1

norm(V) %1

norm(U*V') %1

% Test U*UT, V*VT, U*VT
U*U' % identity

V*V' % Identity

U*V'

imagesc(U*V')

%% SVD, Matrix Inverse and Pseudoinverse

% define the matrix
A = [1 2 3;
     1 2 4;
     1 2 5];

[U,S,V] = svd(A);

% Pseudoinvert S
nonzeroels = S>eps;
S(nonzeroels) = 1./S(nonzeroels);

% Now Pseudoinvert A
Ai = V*S*U';

% Close to I
Ai*A;

% Compare to Matlab's pinv()
[Ai pinv(A)]

pinv(A)*A

%% Condition Number of a Matrix

% define a matrix
m = 40;

% define a 2D Gaussian for smoothing
k = round(m/2);
[X,Y] = meshgrid(linspace(-3,3,k));
g2d = exp(-(X.^2 + Y.^2)/(k/8));

% matrix
A = randn(m);
% A =conv2(randn(m),g2d,"same"); % later comment it out to see the
% difference between random and more structured data

% SVD: We only need the single values
s = svd(A);

% compute the condition number
condnum = s(1)/s(end);

% show the matrix
figure(6), clf
subplot(211), imagesc(A), axis square, axis off
title(["Condition Number: " num2str(condnum)])

subplot(212)
plot(s, "ks-", "linew", 2, "markersize",10,"markerfacecolor","w")
xlabel("Component Number"), ylabel("$\sigma")
title('"Scree Plot" of Singular Values')

%% Code Challenge: Create Matrix with Desired Condition Number
% create a random matrix with a specified condition number
m = 6;
n = 16;

condnum = 41.99;

% create singular vector matrices (orthogoal using QR decomposition)
[U,junk] = qr(randn(m));
[V,junk] = qr(randn(n));
s = linspace(condnum,1,min(m,n));

% create singular value matrix
S = zeros(m,n);
for i=1:length(s)
    S(i,i) = s(i);
end

% Create matrix A
A = U*S*V';
cond(A)

figure(7),clf
subplot(231), imagesc(U), axis square, axis off, title("U")
subplot(232), imagesc(S), axis square, axis off, title("\Sigma")
subplot(233), imagesc(V'), axis square, axis off, title("V^T")
subplot(235), imagesc(A), axis square, axis off 
title("A(Cond.Num=", num2str(cond(A)))

%% Code Challenge: Why you Avoid the Inverse

% Create a matrix with a known condition number
% compute its explicit inverse using inv() function
% Multiply the two to get I
% Compute the norm of the difference between I and eye()
% Repeat for matrices of size 2-70 and condition numbers btw 10-10^12
% Show results in an image

matrixSizes = 2:70;

condnums = linspace(10,1e12,40);

invDifs = zeros(length(matrixSizes),length(condnums));

for mi=1:length(matrixSizes)
    for ci=1:length(condnums)
        
        % create A
        [U,J] = qr(randn(matrixSizes(mi)));
        [V,J] = qr(randn(matrixSizes(mi)));
        S = diag(linspace(condnums(ci),1,matrixSizes(mi)));
        A = U*S*V';

        % Difference with identity matrix
        I = A*inv(A);
        Idiff = abs(I - eye(matrixSizes(mi)));

        % norm of the difference
        invDifs(mi,ci) = norm(Idiff);
    end
end

pcolor(condnums, matrixSizes, invDifs)
xlabel("Condition Number")
ylabel("Matrix Size")
set(gca,"clim",[0 max(invDifs(:))*0.6])
colorbar