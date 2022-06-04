%% LS via row reduction

m = 10;
n = 3;

% create the data
X = randn(m,n); % design matrix
y = randn(m,1); % outcome measures

% apply rref directly
rref([X y]);

% Identity matrix with 4 rows, rest zero

% reaply to the normal equations
Xsol = rref([X'*X X'*y]);

beta = Xsol(:,end)

% compare to left inverse: They are the same
beta2 = (X'*X)\(X'*y)

% Simplified even more (matlab version dependent)
X\y

%% LS Applications 1

% data
data = [-4,0,-3,1,2,8,5,8]';

N = length(data);

% design matrix
X = ones(N,1);

% fit the model
b = (X'*X)\(X'*data);

% compare with the mean
m = mean(data);

[b m] 

% compute model-predicted values
yHat = X*b;

% plot
figure(3), clf
plot(1:N,data,'ks-','markerfacecolor','k','linew',2,'markersize',14)
hold on
plot(1:N,yHat,'ro--','markerfacecolor','r','linew',2,'markersize',14)

set(gca,'xlim',[.5 N+.5])
xlabel('Data point'), ylabel('Data value')
legend({'Observed data';'Predicted data'})

%% Capture the linear trend shown in figure above
% design matrix
X = (1:N)';

% fit the model
b = (X'*X)\(X'*data);

% compare against the mean
m = mean(data);

% compute the model-predicted values
yHat = X*b;

% plot data and model prediction
figure(4), clf
plot(1:N,data,'ks-','markerfacecolor','k','linew',2,'markersize',14)
hold on
plot(1:N,yHat,'ko--','markerfacecolor','k','linew',2,'markersize',14)

set(gca,'xlim',[.5 N+.5])
xlabel('Data point'), ylabel('Data value')
legend({'Observed data';'Predicted data'})

%% Add the Intercept
% design matrix
X = [ ones(N,1) (1:N)' ];

% fit the model
b = (X'*X)\(X'*data);

% compare against the mean
m = mean(data);

% compute the model-predicted values
yHat = X*b;

% plot data and model prediction
figure(5), clf
plot(1:N,data,'ks-','markerfacecolor','k','linew',2,'markersize',14)
hold on
plot(1:N,yHat,'ko--','markerfacecolor','k','linew',2,'markersize',14)

set(gca,'xlim',[.5 N+.5])
xlabel('Data point'), ylabel('Data value')
legend({'Observed data';'Predicted data'})

%% Add nonlinear term by squaring
% design matrix
X = [ ones(N,1) (1:N)'.^2 ];

% fit the model
b = (X'*X)\(X'*data);

% compare against the mean
m = mean(data);

% compute the model-predicted values
yHat = X*b;

% plot data and model prediction
figure(6), clf
plot(1:N,data,'ks-','markerfacecolor','k','linew',2,'markersize',14)
hold on
plot(1:N,yHat,'ro--','markerfacecolor','r','linew',2,'markersize',14)

set(gca,'xlim',[.5 N+.5])
xlabel('Data point'), ylabel('Data value')
legend({'Observed data';'Predicted data'})

%% Least Squares Applications 2
% load the data
load EEG_RT_data.mat

N = length(rts);

% show the data
figure(4), clf
subplot(211)
plot(rts,'ks-','markersize',14,'markerfacecolor','k')
xlabel('Trial'), ylabel('Response time (ms)')

subplot(212)
imagesc(1:N,frex,EEGdata)
axis xy
xlabel('Trial'), ylabel('Frequency')
set(gca,'clim',[-3 3])

%% Model creation and fitting

% design matrix
X = [ ones(N-1,1) rts(1:end-1)' EEGdata(9,2:end)' ];

% compute parameters
b = (X'*X)\(X'*rts(2:end)');

% interpreting the coefficients:
disp([ 'Intercept: ' num2str(b(1)) ' ms' ])
disp([ 'Effect of prev. RT: ' num2str(b(2)) ' ms' ])
disp([ 'Effect of EEG energy: ' num2str(b(3)) ' ms' ])

%% compute effect over frequencies

b = zeros(size(frex));

for fi=1:length(frex)
    
    % design matrix
    X = [ ones(N,1) EEGdata(fi,:)' ];
    
    % compute parameters
    t = (X'*X)\(X'*rts');
    b(fi) = t(2);
end

% plot
figure(5), clf
subplot(211)
plot(frex,b,'rs-','markersize',14,'markerfacecolor','k')
xlabel('Frequency (Hz)')
ylabel('\beta-coefficient')


% scatterplots at these frequencies
frex2plot = dsearchn(frex',[ 8 20 ]');

for fi=1:2
    subplot(2,2,2+fi)
    
    plot(EEGdata(frex2plot(fi),:),rts,'rs','markerfacecolor','k')
    h=lsline;
    set(h,'linew',2,'color','k')
    
    xlabel('EEG energy'), ylabel('RT')
    title([ 'EEG signal at ' num2str(round(frex(frex2plot(fi)))) ' Hz' ])
end

%% Code Challenge: LS via QR Decomposition
% generate random data
m = 10;
n = 3;

X = randn(m,n);
y = randn(m,1);

% Solve for bet using QR

[Q,R] = qr(X);

beta = (R'*R)\(Q*R)'*y;

% compare QR with standard left-inverse method
beta2 = X\y;

[beta beta2]


