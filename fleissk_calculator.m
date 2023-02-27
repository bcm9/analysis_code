function [fleissk] = fleissk_calculator(n,N,k,k_table)
% fleissk_calculator Fleiss' kappa calculation for interrater agreement of categorical ratings
% Arguments: fleissk_calculator(n,N,k)
%   n = raters
%   N = subjects
%   k = categories
%   k_table = N x k matrix
% example from https://en.wikipedia.org/wiki/Fleiss%27_kappa:
% k_table=[0 0 0 0 14;0 2 6 4 2; 0 0 3 5 6; 0 3 9 2 0; 2 2 8 1 1;7 7 0 0 0; 3 2 6 3 0; 2 5 3 2 2; 6 5 2 1 0 ;0 2 2 3 7;];
% [fleissk]=fleissk_calculator(14,10,5,k_table)
% https://github.com/bcm9

% pj
p_j=sum(k_table)./sum(sum(k_table));
% Pi
for nk=1:N
    P_i(nk)=1/(n*(n-1))*(sum(k_table(nk,:).^2)-n);
end
% P_hat = sum of Pi /N
P_hat=sum(P_i)*1/N;
% P_hat_e = sum of pj^2
P_hat_e=sum(p_j.^2);
% fleiss k = P_hat - P_hat_e / 1-P_hat_e
fleissk=(P_hat-P_hat_e)/(1-P_hat_e);
end
