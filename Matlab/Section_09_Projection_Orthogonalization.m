%% Projections in R2

% define a point b
b = [4 1]';

% define line a
a = [2 5]';

% beta
beta = (a'*b)/(a'*a);

figure(1), clf
plot(b(1), b(2), "ko", "markerfacecolor", "m", "markersize", 20)
hold on
plot([0, a(1)], [0, a(2)], "b", "linew", 3)

% plot the projection line
plot([b(1) beta*a(1)], [b(2) beta*a(2)], "r--", "linew",3)

legend({'b';'a';'b-\betaa'})
axis([-1 1 -1 1]*max([norm(a) norm(b)]))
plot(get(gca,'xlim'),[0 0],'k--')
plot([0 0],get(gca,'xlim'),'k--')
axis square

%% Projections in R^N
% Solve Ax = b for x

m = 16;
n = 10;

% vector b
b = randn(m,1);

% Matrix A
A = randn(m,n);

% Solution using explicit inverse
x1 = inv(A'*A) * (A'*b);

% Preferred solution
x2 = (A'*A)\ (A'*b);

% Another possibility (doesnt work in older versions of matlab)
x3 = A\b;

% They are all the same

%% Geomteric Perspective
m = 3;
n = 2;

b = randn(m,1);
A = randn(m,n);

% solution for x
x = (A'*A) * (A'*b);

Ax = A*x;

figure(2), clf
h = ezmesh( @(s,t)A(1,1)*s+A(1,2)*t , @(s,t)A(2,1)*s+A(2,2)*t , @(s,t)A(3,1)*s+A(3,2)*t , [-1 1 -1 1]*norm(x)*2);
set(h,'facecolor','g','cdata',ones(50),'EdgeColor','none')
xlabel('R_1'), ylabel('R_2'), zlabel('R_3')
axis square
grid on, rotate3d on, hold on

h(1) = fplot3( @(t)t*b(1), @(t)t*b(2), @(t)t*b(3) , [-1 1]);
h(2) = fplot3( @(t)t*Ax(1), @(t)t*Ax(2), @(t)t*Ax(3) , [-1 1]);
plot3(Ax(1),Ax(2),Ax(3),'ro','markerfacecolor','r','markersize',12)

set(h,'LineWidth',3)
title('')
legend({'C(A)';'b';'Ax'})

%% Vector Decomposition
% w: to be decomposed
w = [2 3]';

% v: reference
v = [4 0]';

% compute w parallel to v
beta = (w'*v)/(v'*v);
w_vp = beta*v;

% compute w orthogonal to v
w_vo = w - w_vp;

% compute results algebraically (orthogonal components' dot product should be zero)
(w_vp + w_vo) - w % should be zero
w_vp'*w_vo;

% plot all 4 vectors
figure(1), clf, hold on
plot([0, w(1)], [0, w(2)], "r", "linew",3)
plot([0, v(1)], [0, v(2)], "b", "linew",2)
plot([0, w_vp(1)], [0, w_vp(2)], "r--", "linew",3)
plot([0, w_vo(1)], [0, w_vo(2)], "r:", "linew",3)

legend({"w"; "v"; "w_{||}v"; "w_{\perp}v"})
axis([-1 1 -1 1]*5)
axis square
grid on

%% QR Decomposition

A = [1 0;
     1 0;
     0 1];

% full QR Decomposition
[Q R] = qr(A);

% Economy QR Decomposition
[Q R] = qr(A,0);

% Another example of "to-be decomposed" matrix
M = [1 2 -2;
     3 -1 1];

% QR decomposition
[Q R] = qr(M);

% Notice: R = Q'*M
Q'*M

% plot
figure(4), clf
colorz = 'krg';

for i=1:size(M,2)
    
    % plot original vector M
    plot([0 M(1,i)],[0 M(2,i)],colorz(i),'linew',3), hold on
    
    % plot orthogonalized vector Q
    if i<=size(Q,2)
        plot([0 Q(1,i)],[0 Q(2,i)],[colorz(i) '--'],'linew',2)
    end
    
    % plot residual vector R
    plot([0 R(1,i)],[0 R(2,i)],[colorz(i) ':'],'linew',2)
end
legend({'M_1';'Q_1';'R_1'})


axis([-1 1 -1 1]*norm(M))
axis square
grid on

%% Code Challenge: Gram-Schmidt Algorithm

% start with a square matrix, compute Q
% Check Q'*Q = I
% check qr
% There may be some sign differences, thats normal
% Extend the code to rectangular matrices

m = 4;
n = 4;

A = randn(m,n);
Q = zeros(m,n);

% loop over columns (n)

for i=1:n

    Q(:,i) = A(:,i);

    a = A(:,i);

    % orthogonalze the ith column in Q relative to previous columns
    for j=1:i-1

        q = Q(:,j);
        Q(:,i) = Q(:,i) - ((a'*q)/(q'*q))*q;
    end

    % normalize the ith column of Q
    Q(:,i) = Q(:,i)/norm(Q(:,i));
end

imagesc(Q'*Q)

%% Code Challenge: Inverse via QR

% Generate a large matrix
% inverse using inv() and QR Decomposition
m = 100;

A = randn(m);

% Compute inverse using inv
Ai = inv(A);

% Inverse via QR
[Q,R] = qr(A);

% R is expected to be an upper triangular matrix
imagesc(R);

AiQR1 = inv(R)*Q';
AiQR2 = R\Q'; % better

subplot(121)
imagesc(AiQR1)
subplot(122)
imagesc(AiQR2)

% See if matrices have the same elements
% flatten all matrices to create a 100x3 matrix and correlate them
corr([ Ai(:) AiQR1(:) AiQR2(:)]);

%% The SHerman-Morrison Formula
m = 5;
a = randn(m,1);
b = randn(m,1);

% create a failure condition (no identity matrix)
% may get inf and NaN, or very small values because of rounding errors
%a = a/norm(a);
%b = a;

A = eye(m) - a*b';
Ai = eye(m) + (a*b')/(1-a'*b);

% Their product should give the identity matrix
A*Ai

%% Code Challenge: Prove that A'*A = R'*R
% 1) Generate a random matrix A
% 2) Compute its QR Decomposition
% 3) test the claim
% 4) Use pen and paper

m = 10;
n = 14;
A = randn(m,n);

[Q,R] = qr(A);

% difference should be zero
A'*A - R'*R
