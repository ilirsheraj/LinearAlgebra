%% Finding Eiganvalues

A = [1 5;
     2 4];

% extract the eiganvalues
eigvals = eig(A);

% specify two vectors
v1 = [1 1]'; % eiganvector of A
v2 = randn(2,1); % unlikely to be an eiganvector
v2 = v2/norm(v2); % unit length for convenience

% compute Av
Av1 = A*v1;
Av2 = A*v2;

% plot the vectors and Av
figure(1), clf
plot([0 v1(1)],[0 v1(2)],'r','linew',4)
hold on
plot([0 Av1(1)],[0 Av1(2)],'r--','linew',2)
plot([0 v2(1)],[0 v2(2)],'k','linew',4)
plot([0 Av2(1)],[0 Av2(2)],'k--','linew',2)
legend({'v';'Av';'w';'Aw'})

lim = max([Av1(:); Av2(:)])*1.2;
axis([-1 1 -1 1]*lim)
grid on, axis square
plot(get(gca,'xlim'),[0 0],'k')
plot([0 0],get(gca,'ylim'),'k')

%% eigenvalues of a 3x3 matrix

% specify matrix used in lecture
A = [ -2  2 -3 ;
      -4  1 -6 ;
      -1 -2  0 ];

% get eigenvalues to confirm our results
eig(A)

%% Code Challenge: eiganvalues of diagonal and triangular matrices

% generate a diagonal matrix (2x2) and comoute its eiganvaues

% expand it to a larger diagonal square matrix

% generate triangular matrices and see eiganvalue patterns for these
% matrices

A = diag([2 1]);

eig(A);

A2 = diag([randn(2,1)]);
eig(A2);

% expand
A = diag([1:10]);
eig(A);

% lower triangular matrix
B = tril(randn(4));
eig(B);

% Upper triangular matrix
C = triu(randn(4));
eig(C)

%% Code Challenge: Eiganvalues of random matrices

% generate a 40x40 random matrix, extract the eiganvalues and plot
% repeat this multiple times and put the eiganvalues in the same plot
itern = 200;
m = 45;
evals = zeros(itern,m);

figure(1), clf, hold on
for i=1:itern
    A = randn(n)/sqrt(m);

    plot(eig(A), "s")
end

%% Finding Eiganvectors

A = [1 2; 2 1];

[evec, evals] = eig(A); % evec = v, evals = d sometimes

% convert eiganvalues to vector
evals = diag(evals); % this is cool

% compute the norm of each eigenvector to see its indeed 1
mag_v1 = sqrt( sum(evec(:,1).^2) );
mag_v2 = sqrt( sum(evec(:,2).^2) );


% plot
figure(2), clf
plot([0 evec(1,1)],[0 evec(2,1)],'r','linew',3)
hold on
plot([0 evec(1,2)],[0 evec(2,2)],'k','linew',3)
legend({'v_1';'v_2'})

axis([-1 1 -1 1])
grid on, axis square
plot(get(gca,'xlim'),[0 0],'k')
plot([0 0],get(gca,'ylim'),'k')

%% Diagonalization

A = round(10*randn(4));

% create symmetric matrix (for visualization and has real value eiganvalues
A = A'*A;

% eignadecomposition
[evecs, evals] = eig(A);

% test reconstruction
Ap = evecs * evals * inv(evecs);

% plot
figure(4), clf
subplot(121), imagesc(A)
axis square, axis off, title("A")

subplot(122), imagesc(Ap)
axis square, axis off, title("VAV^{-1}")

%% Matrix powers via diagonalization

A = randn(2);

A3 = A*A*A;

% matrix power via eigandecomposition

[V,D] = eig(A);

A3e = V* D^3 * inv(V);

[A3 A3e] % same

%% Eigandecomposition of A^3

A = randn(3);

A = A'*A;

% matrix power via eigandecomposition

[V,D] = eig(A);

[V3,D3] = eig(A^3);

% Sort the results
[d,sidx] = sort(diag(D), "descend");
V = V(:,sidx);
D = diag(d);

[d,sidx] = sort(diag(D3), "descend");
V3 = V3(:,sidx);
D3 = diag(d);

figure(6), clf
subplot(221), imagesc(V)
axis square, title("evecs of A")

subplot(223), imagesc(V3)
axis square, title("evecs of A^3")

% plot eigenvectors of A
subplot(222), hold on
plot3([0 V(1,1)],[0 V(2,1)],[0 V(3,1)],'r','linew',3)
plot3([0 V(1,2)],[0 V(2,2)],[0 V(3,2)],'k','linew',3)
plot3([0 V(1,3)],[0 V(2,3)],[0 V(3,3)],'b','linew',3)
axis([-1 1 -1 1 -1 1]), axis square
rotate3d on, grid on

% plot eigenvectors of A^3
plot3([0 V3(1,1)],[0 V3(2,1)],[0 V3(3,1)],'r--','linew',3)
plot3([0 V3(1,2)],[0 V3(2,2)],[0 V3(3,2)],'k--','linew',3)
plot3([0 V3(1,3)],[0 V3(2,3)],[0 V3(3,3)],'b--','linew',3)
title('Eigenvectors')


subplot(224)
plot(1:3,diag(D),'bs-','linew',3,'markersize',15,'markerfacecolor','w')
hold on
plot(1.1:3.1,diag(D3),'rs-','linew',3,'markersize',15,'markerfacecolor','w')
set(gca,'xlim',[.5 3.5]), axis square
title('Eigenvalues')
legend({'A';'A^3'})

%% Eigandecomposition of Matrix differences
A = randn(5);
%A = A'*A;

B = randn(5);
%B = B'*B;

[V,D] = eig(A-B);

[V2,D2] = eig(A^2 -A*B -B*A + B^2); % orders are different

[d1, sidx] = sort(diag(D), "descend");
v1 = V(:,sidx);

[d2, sidx] = sort(diag(D2), "descend");
v2 = V2(:,sidx);

[d1, d2]

% for non-symmetric comment out and see what happens

%% Eiganvectors of Repeated Eiganvalues

% define a matrix
A = [5 -1 0;
     -1 5 0;
     1/3 -1/3 4];

% perform eigandecomposition
[V,D] = eig(A);

% sort the eiganvalues
[D, sidx] = sort(diag(D));
V = V(:,sidx);

% plot eigenvectors
figure(2), clf, hold on
plot3([0 V(1,1)],[0 V(2,1)],[0 V(3,1)],'k','linew',3)
plot3([0 V(1,2)],[0 V(2,2)],[0 V(3,2)],'r','linew',3)
plot3([0 V(1,3)],[0 V(2,3)],[0 V(3,3)],'b','linew',3)
legend({[ 'v_1 (\lambda=' num2str(D(1)) ')' ];[ 'v_1 (\lambda=' num2str(D(2)) ')' ];[ 'v_3 (\lambda=' num2str(D(3)) ')' ]})

% plot subspace spanned by same-eigenvalued eigenvectors
h = ezmesh( @(s,t)V(1,1)*s+V(1,2)*t , @(s,t)V(2,1)*s+V(2,2)*t , @(s,t)V(3,1)*s+V(3,2)*t , [-1 1 -1 1 -1 1]);
set(h,'facecolor','g','cdata',ones(50),'LineStyle','none')
xlabel('eig_1'), ylabel('eig_2'), zlabel('eig_3')
axis square, grid on, rotate3d on
title('')

%% Eigandecomposition of Symmetric Matrices

% Create a random matrix
A = randn(14);

% Make it symmetric using the additive method
A = A'*A;

% Decompose it
[V,D] = eig(A);

% Magnitude of each vector
sqrt(sum(V.^2,1));

figure(6), clf
subplot(131), imagesc(A)
axis square, axis off
title("A")

subplot(132), imagesc(V)
axis square, axis off
title("Eiganvectors")

subplot(133), imagesc(V'*V)
axis square, axis off
title("VV'")

%% Code Challenge: Reconstruct matrix from Eiganlayers

% create an mxm symmetric matrix and take its eigandecomposiion
% show that the norm of the outer product of v_i
% create one layer of A as lambda*v*v'
% compute its norm and test whether the norm is the eiganvaliu
% reconstruct A by summing over the layers

m = 5;
A = randn(m);
A = round(10*A'*A);

% eig of A
[V,D] = eig(A);

% Check V'*V (should be I)

% Compute norom of outer product v_i
V(:,3)*V(:,3)'; % gives a matrix of rank 1

norm( V(:,3)*V(:,3)'); % it is 1

% create one layer of A as lambda*v*v'
D(3,3); % third eiganvalue
norm( V(:,3)* D(3,3)*V(:,3)'); % norm equals wiganvalue

rank( V(:,3)* D(3,3)*V(:,3)'); % rank is still 1

% reconstruct A by summing over the layers
Arecon = zeros(m);

for i=1:m
    Arecon = Arecon + V(:,i)* D(i,i)*V(:,i)'
    disp(rank(Arecon))
end

%% Trace and determinant, Eiganvalues sum and product
% trace(A) = sum(evals)
% det(A) = prod(evals)

% first use a full-rank, then a reduced-rank matrix

A = randn(7);

trA = trace(A);
dtA = det(A);

% decomposition
l = eig(A);

[trA sum(l)]

[dtA prod(l)]

% Repeat the same thing for a reduced rank matrix
A = randn(7,5) * randn(5,7);

trA = trace(A);
dtA = det(A);

% decomposition
l = eig(A);

[trA sum(l)]

[dtA prod(l)] % they are zero as well

%% Generalized Eigandecomposition
A = [3 2; 1 3];
B = [1 1; 4 1];

% GED
[eigvecs, eigvals] = eig(A,B); % 2 matrices, order important

% matrix-vector multiplication
Av = A*eigvecs(:,2);
Bv = B*eigvecs(:,2);
BinvAv = inv(B)*A*eigvecs(:,2);

figure(1), clf
subplot(131)
plot([0 eigvecs(1,2)],[0 eigvecs(2,2)],'k','linew',4), hold on
plot([0 Av(1)],[0 Av(2)],'r--','linew',2)
axis square, axis([-1 1 -1 1]*3), plot(get(gca,'xlim'),[0 0],'k:'), plot([0 0],get(gca,'ylim'),'k:')
legend({'v_2';'Av_2'})
title('Av')


subplot(132)
plot([0 eigvecs(1,2)],[0 eigvecs(2,2)],'k','linew',4), hold on
plot([0 Bv(1)],[0 Bv(2)],'r--','linew',2)
axis square, axis([-1 1 -1 1]*3), plot(get(gca,'xlim'),[0 0],'k:'), plot([0 0],get(gca,'ylim'),'k:')
legend({'v_2';'Bv_2'})
title('Bv')


subplot(133)
plot([0 eigvecs(1,2)],[0 eigvecs(2,2)],'k','linew',4), hold on
plot([0 BinvAv(1)],[0 BinvAv(2)],'r--','linew',2)
axis square, axis([-1 1 -1 1]*3), plot(get(gca,'xlim'),[0 0],'k:'), plot([0 0],get(gca,'ylim'),'k:')
legend({'v_2';'B^{-1}Av_2'})
title('B^-^1Av')

%% Code Challenge: GED in small and large matrices
% goal 1: compare eig(S,R) with eig(inv(R)*S)
% part 1: GED on 2x2 matrix
% part 2: real data matrices

S = randn(2);
R = randn(2);

[Ws, Ls] = eig(S,R);
[Wi, Li] = eig(inv(R)*S);

[diag(Ls) diag(Li)]; % same

figure(1), clf, hold on
plot([0 Ws(1,1)], [0 Ws(2,1)], "k-")
plot([0 Ws(1,2)], [0 Ws(2,2)], "k--")
plot([0 Wi(1,1)], [0 Wi(2,1)], "r-")
plot([0 Wi(1,2)], [0 Wi(2,2)], "r--")
axis([-1 1 -1 1]*1.5), axis square, grid on

%% part 2: real data matrices
load real_matrices.mat
[Ws, Ls] = eig(S,R); % The correct way to do things
[Wi, Li] = eig(inv(R)*S); % not stable, untrustable results

[rank(S) rank(R)];

plot(diag(Ls), "s-") % eiganspectrum
hold on
plot(real(diag(Li)), "r-") % The results are very different because its singular matrix

imagesc(real(Wi))

