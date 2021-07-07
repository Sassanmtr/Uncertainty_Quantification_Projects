%====================================
% Weibull Distribution
%====================================
M = 1e5;
N = 5:12;
gammav = 0.01;
lambda = 1;       % scale parameter
k = 0.5;          % shape parameter
%------------------------------------
CV_ME = zeros(1,length(N));
CV_KL = zeros(1,length(N));
alpha_ME = zeros(1,length(N));
alpha_KL = zeros(1,length(N));
for i = 1:length(N)
    [alpha_ME(i),CV_ME(i)] = maxent(M,N(i),gammav,lambda,k);
    [alpha_KL(i),CV_KL(i)] = kldiv(M,N(i),gammav,lambda,k); 
end
CV_SR = (((1-exp(-(gammav/lambda)^k)).^N)./alpha_ME)-1;

figure
h = semilogy(N , CV_SR, 'g', N, CV_ME, 'b', N, CV_KL, 'r');
legend('IS wrt \gamma','Max Entropy', 'Proposed IS', 'location','northwest')
xlabel('N')
ylabel('Squared Coefficient of Variation')
set(gca,'fontsize', 16);
set(h(1),'linewidth',2);
set(h(2),'linewidth',2);
set(h(3),'linewidth',2);
saveas(gcf,'exp2.png')

function [alpha,CV] = kldiv(M, N, gammav, lambda, k) 
    theta = -(N/gammav)*k;
    gamma_fun = gamma(k);
    M_tilde = gamma_fun/(-theta)^k;
    X = gamrnd(k,-1/theta,[N,M]);
    Y = sum(X) <= gammav;
    f_old = (k/lambda) *  (X./lambda).^(k-1) .* exp(-(X/lambda).^k);
    f_new = ((X.^(k-1)) .* exp(theta*X))/M_tilde;
    L = prod(f_old) ./ prod(f_new);
    alpha = mean(Y .* L);
    L_CV = L.^2;
    CV = (mean(Y .* L_CV)/(alpha^2))-1;
end

function [alpha, CV] = maxent(M, N, gammav, lambda, k) 
    X = (-1/N)*log(1-(1-exp(-N))*rand(N,M)); 
    Y = sum(X) <= 1;
    f_old = 1/(1-exp(-(gammav/lambda)^k))*((k*gammav)/lambda) * (X./(lambda/gammav)).^(k-1) .* exp(-(X./(lambda/gammav)).^k);
    f_new = (N*exp(-N*X)) ./ (1-exp(-N));
    L = prod(f_old) ./ prod(f_new);
    MC = mean(Y .* L);
    cdf = (1-exp(-(gammav/lambda)^k))^N;
    alpha = MC * cdf;
    L_CV = prod(f_old .^ 2) ./ prod(f_new .^ 2);
    CV = (((cdf^2) * mean(Y .* L_CV)) / (alpha^2))-1;
end

