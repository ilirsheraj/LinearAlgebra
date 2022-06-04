%% Creating and plotting vectors

% A 2D Row vector

v2 = [3 -2];

% 3D row vector

v3 = [4 -3 2];

% Row to columns and vice versa

v3'

% Plot them

figure(1), clf
subplot(211)

% The main plotting line

plot([0 v2(1)], [0 v2(2)], "Linew",2)

% Make the plot look better
axis square
axis([-4 4 -4 4])
hold on
plot(get(gca, "xlim"), [0 0], "k--")
plot([0 0 ], get(gca, "ylim"), "k--")
xlabel("X_1 Dimension")
ylabel("X_2 Dimension")

% A 3D vector
subplot(212)

% The main plotting code
plot3([0 v3(1)], [0 v3(2)], [0 v3(3)], "linew", 2)
axis square
axis([-4 4 -4 4 -4 4])
hold on, grid on
plot3(get(gca, "xlim"), [0 0], [0 0], "k--")
plot3([0 0 ], get(gca, "ylim"), [0 0 ], "k--")
plot3([0 0 ], [0 0], get(gca, "zlim"), "k--")
xlabel("X_1 Dimension")
ylabel("X_2 Dimension")
zlabel("X_3 Dimension")

% rotate
rotate3d on

%% Section 2: Vector Addition and Subtraction

v1 = [3 -1];
v2 = [2 4];
v3 = v1 + v2;

% Plot them
figure(2), clf
% The main plotting lines
plot([0, v1(1)], [0, v1(2)], "b","linew", 2)
hold on
plot([0, v2(1)] + v1(1), [0, v2(2)] + v1(2), "r","linew", 2)
plot([0, v3(1)], [0, v3(2)], "k","linew", 3)
legend(["v1"; "v2"; "v1 + v2"])

% this code just makes the plot look nicer
axis square
axis([ -1 1 -1 1 ]*6)
hold on
plot(get(gca,'xlim'),[0 0],'k--')
plot([0 0],get(gca,'ylim'),'k--')
xlabel('X_1 dimension')
ylabel('X_2 dimension')

%% Section 3: Vector-Scalar Multiplication

v1 = [3 -1];
l = -.3;

%plot them
figure(3), clf

plot([0, v1(1)], [0, v1(2)], "b","linew", 2)
hold on
plot([0, v1(1)]*l, [0, v1(2)]*l, "r", "linew", 4)
legend(["v1"; "v2"])

% Better Visualization
axis square
axis([-1 1 -1 1]*max([norm(v1) norm(v1*l)]))
hold on
plot(get(gca,'xlim'),[0 0],'k--')
plot([0 0],get(gca,'ylim'),'k--')
xlabel('X_1 dimension')
ylabel('X_2 dimension')

%% Dot Product
% There are 4 different wayt to compute dot product in matlab

v1 = [1 2 3 4 5];
v2 = [0 -4 -3 6 5];

% method 1
dp = sum(v1.*v2);

% method 2
dp = dot(v1, v2);

% method 3
dp = v1*v2';

% method 4
dp = 0; %initialize
% loop over elements
for i = 1:length(v1)

    % multiply corresponding elements and sum
    dp = dp + v1(i)*v2(i);
end
dp

%% Distributive Properties of Vector dot product

n = 10;
a = randn(n,1);
b = randn(n,1);
c = randn(n,1);

% The two results
res1 = a' * (b+c);
res2 = a'*b + a'*c;

% compare the results
disp([res1 res2])

%% Associative Properies of Vector Dot Product
n = 10;
a = randn(n,1);
b = randn(n,1);
c = randn(n,1);

% The two results
res1 = a' * (b'*c);
res2 = (a'*b)'*c;

% transpose res1 for comparison purposes
disp([res1' res2])

% If dimesnions are not the same it will not work

v1 = [1 2 3 4 5];
v2 = [1 2 3 4 5];
v3 = [1 2 3 4 5];

res1 = v1 * dot(v2, v3);
res2 = v2 * dot(v1, v3);
res3 = v3 * dot(v1, v2);

disp([res1' res2' res3'])

%% Random matrices

A = randn(4, 6);
B = randn(4, 6);

dps = zeros(size(A,2), 1);
for i = 1:size(A,2)
    dps(i) = dot(A(:,i), B(:,i));
end

disp(dps)

%% Is Dot Product Commutative
% a'*b == b'*a

a = randn(1, 100);
b = randn(1, 100);

res1 = a* b';
res2 = b *a';

disp([res1 res2])

% Generate a two 2-element integer vectors

a = [2 4];
b = [3 5]

[a*b' b*a']

%% Vector Length
v1 = [1 2 3 4 5 6];

% method 1
m1 = sqrt(sum(v1.*v1));

% method 2
m2 = norm(v1);

disp([m1 m2])

%% Dot Product from a Geometric Perspective

% two vectors
v1 = [ 2  4 -3 ];
v2 = [ 0 -3 -3 ];

% compute the angle (radians) between two vectors
ang = acos( dot(v1,v2) / (norm(v1)*norm(v2)) );


% draw them
figure(4), clf
plot3([0 v1(1)],[0 v1(2)],[0 v1(3)],'b','linew',2)
hold on
plot3([0 v2(1)],[0 v2(2)],[0 v2(3)],'r','linew',2)

axmax = max([ norm(v1) norm(v2) ]);
axis([-1 1 -1 1 -1 1]*axmax)
grid on, rotate3d on, axis square
title([ 'Angle between vectors: ' num2str(rad2deg(ang)) '^0' ])

%% equivalence of algebraic and geometric dot product formulas

% two vectors
v1 = [ 2  4 -3 ];
v2 = [ 0 -3 -3 ];


% algebraic
dp_a = sum( v1.*v2 );

% geometric
dp_g = norm(v1)*norm(v2)*cos(ang);

% print dot product to command
disp([ 'Algebra: ' num2str(dp_a) ', geometry: ' num2str(dp_g) ])

%% The C-S Theorem

a = randn(5, 1);
b = randn(5, 1);
c = randn*a;

% dot products: LHS of the equation
aTb = dot(a, b);
aTc = dot(a,c);

clc
disp([abs(aTb) norm(a)*norm(b)])
disp([abs(aTc) norm(a)*norm(c)])

%% Soft Proof
% Soft proof is never like a rigorous mathematical proof, but they help to
% test multiple options
% Is dot product invariant to scalar multiplications

% Generate two 3D vectors (R3)
%v1 = [-3 4 5]';
%v2 = [3 6 -3]';

v1 = [-3 4 6]';
v2 = [3 6 -3]';

% Generate two scalars
s1 = 2;
s2 = 3;

% compute dot product between vectors
disp(["original: " num2str(dot(v1, v2))])

% compute dot product between scaled vectors
disp(["scaled: " num2str(dot(s1*v1, s2*v2))])

disp(["scaled: " num2str( (s1*v1)' * (s2*v2))])

%% Hadamard Multiplication

w1 = [1 3 5];
w2 = [3 4 2];

w3 = w1.* w2;
w3

%% Outer Product
% Transpose to make them column vectors
v1 = [1 2 3]';
v2 = [-1 0 1]';

% dot product
v1' * v2

% outer product
v1 * v2'

% conceptual inplementation
op = zeros(length(v1), length(v2));
for i = 1:length(v1)
    for j = 1:length(v2)
        op(i,j) = v1(i) * v2(j);
    end
end

%% Cross-Product
v1 = [-3 2 5];
v2 = [4 -3 0];

% Matlab's cross-product function
v3a = cross(v1, v2);

% manual method
v3b = [ v1(2) * v2(3) - v1(3) * v2(2) ;
    v1(3) * v2(1) - v1(1) * v2(3) ;
    v1(1) * v2(2) - v1(2) * v2(1)];

figure(5), clf, hold on
h = ezmesh( @(s,t)v1(1)*s+v2(1)*t , @(s,t)v1(2)*s+v2(2)*t , @(s,t)v1(3)*s+v2(3)*t , [-1 1 -1 1]*2);
set(h,'facecolor','g','cdata',ones(50),'EdgeColor','none')

% individual vectors
plot3([0 v1(1)],[0 v1(2)],[0 v1(3)],'k','linew',3)
plot3([0 v2(1)],[0 v2(2)],[0 v2(3)],'k','linew',3)
plot3([0 v3a(1)],[0 v3a(2)],[0 v3a(3)],'r--','linew',3)

% make the plot look a bit nicer
xlabel('R_1'), ylabel('R_2'), zlabel('R_3')
axis square
grid on, rotate3d on, hold on
title('')

%% Hermitian Transpose

% create a complex number
z = complex(3, 4);

% calculate its magnitude
norm(z)

% normal dot product
transpose(z)*z

% hermitian transpose
% z' gives us the complex number with negative imaginary component
z' * z

% not the hermitian
% if we put dot before transpose, no negative conjugate is created, so it
% will be like normal transpose
z.' * z

% complex vector
% matlab converts it into complex vector automatically
v = [3 4i 5+2i complex(2, -5)];

% with appostrophe it will be converted into hermitian automatically
% with dot asterisks it will not flip the sign
% with transpose() function it will also not flip the sign
transpose(v);

%% Unit Vectors
v1 = [-3 6];

% mu
mu = 1/norm(v1);

% plot them
figure(5), clf

% the main plotting lines
plot([0 v1(1)], [0 v1(2)], "b","linew" ,2)
hold on
plot([0 v1(1)*mu], [0 v1(2)*mu], "r", "linew", 4)
legend({"v1"; "v1-unit"})

% To make the plot look nicer
axis square
axis([-1 1 -1 1]*norm(v1))
hold on
plot(get(gca,'xlim'),[0 0],'k--')
plot([0 0],get(gca,'ylim'),'k--')
xlabel('X_1 dimension')
ylabel('X_2 dimension')

%% Coding Challenge

% Create two random integer vectors (R4)
n = 4;
v1 = round(20*randn(n,1));
v2 = round(20*randn(n,1));

% Compute the lengths of individual vectors and the dot products
v1l = norm(v1);
v2l = norm(v2);
v1d2 = dot(v1l, v2l);

% Normalize the vectors (create unit vectors)
v1n = v1/norm(v1);
v2n = v2/norm(v2);

% Compute the magnitude of dot product of normalized vectors
v2d2n = dot(v1n, v2n);
disp([v1d2 v2d2n])

%% Span

% Define set S
S = [[1 1 0]' [1 7 0]'];

% vectors v and w
v = [1 2 0];
w = [3 2 1];

figure(6), clf, hold on
plot3([0 S(1,1)],[0 S(2,1)],[0 S(3,1)],'g','linew',3)
plot3([0 S(1,2)],[0 S(2,2)],[0 S(3,2)],'g','linew',3)

plot3([0 v(1)],[0 v(2)],[0 v(3)],'k','linew',3)
plot3([0 w(1)],[0 w(2)],[0 w(3)],'b','linew',3)

% draw the plane spanned by S
normvec = cross(S(:,1),S(:,2));
[X,Y] = meshgrid(-4:4,-4:4);
z = -(normvec(1)*X + normvec(2)*Y)/normvec(3);
surf(X,Y,z)
shading interp

axis square
grid on, rotate3d on