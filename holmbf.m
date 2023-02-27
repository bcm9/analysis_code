function [ output_args ] = holmbf(pvals, alpha)
%holmbf Holm-Bonferroni Method for Multiple Comparisons
%   Returns matrix of pvals, hb alphas, and whether to reject null for array of p-values
%   e.g. holmbf([0.01 0.04 0.03 0.005], 0.05)

% set target alpha to 0.05 if not set
if nargin <2
    alpha=0.05;
end

% order p-values from smallest to largest
pvals=sort(pvals);

for rank=1:length(pvals)
    % calculate new hb alpha
    hbalpha(rank)=alpha/(length(pvals)-rank+1);
    % reject the null if the pvalue is less than the hb alpha
    rejnull(rank)=pvals(rank)<hbalpha(rank);
    % if fail to reject null, stop loop
    if rejnull(rank)==0
        break
    end
end

disp('pvals, hbalpha, reject null (1=yes)?')
% display pvalues, new alphas up to non-rejection (0)
[pvals(1:length(rejnull))' hbalpha(1:length(rejnull))' rejnull']
end
