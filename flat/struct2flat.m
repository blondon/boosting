function [X,Y] = struct2flat(examples)

% Calculate number of observations
nObs = 0;
for i = 1:length(examples)
	nObs = nObs + length(examples(i).Y);
end

% Allocate outputs
X = zeros(nObs,size(examples(1).Xnode,2));
Y = zeros(nObs,1);

% Fill outputs
filled = 0;
for i = 1:length(examples)
	nObs_i = length(examples(i).Y);
	X(filled+1:filled+nObs_i,:) = squeeze(examples(i).Xnode(1,:,:))';
	Y(filled+1:filled+nObs_i) = examples(i).Y;
	filled = filled + nObs_i;
end

% For liblinear (comment out for libsvm)
X = sparse(X);
Y = sparse(Y);
