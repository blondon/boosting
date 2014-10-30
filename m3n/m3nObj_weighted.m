function [f,g,psi,ham] = m3nObj_weighted(w, examples, lambda, decodeFunc, dist)
%
% Outputs the objective value and gradient of the M3N learning objective
%
% w : current weights
% examples : array of example structure, each containing:
%	ss : nParam x 1 vector of sufficient statistics
%	Ynode : nState x nNode overcomplete matrix representation of labels
% lambda : regularization constant or vector
% decodeFunc : decoder function
% dist : distribution over examples and labels

% Initialization
nEx = length(examples);
psi = zeros(size(w));
ham = 0;

% Loop over examples
for i = 1:nEx

	ex = examples(i);
	nNodes = double(ex.edgeStruct.nNodes);
	d = dist{i};
	
	% Normally, the loss-augmented potentials would be scaled by the
	% distribution; i.e.,
	%   nodePot = nodePot.^(sum(d)/nNodes) .* exp(bsxfun(@times,1-ex.Ynode',d));
	% However, this might cause numerical instability. Instead, we
	% scale the loss-augmentation by (nNodes / sum(d)).
	C = nNodes / sum(d);
	
	% Loss-augmented inference
	[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,ex.edgeStruct);
	nodePot = nodePot .* exp(bsxfun(@times,1-ex.Ynode',C*d));
	yhat = decodeFunc(nodePot,edgePot,ex.edgeStruct);

	% Compute sufficient statistics
	ss_mu = sufficientStats(yhat,ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,ex.edgeStruct);

	% ham_i = sum_j Hamming(y_i(j), yhat_i(j)) * P(i,j)
	% psi_i = (ss_yhat_i - ss_y_i) * P(i) / nNodes
	ham_i = (ex.Y ~= yhat)' * d;
	psi_i = (ss_mu - ex.ss) * (sum(d) / nNodes);
	ham = ham + ham_i;
	psi = psi + psi_i;

end

% Objective
f = 0.5*w'*(lambda.*w) + w'*psi + ham;

% Gradient
g = lambda.*w + psi;
