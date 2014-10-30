function [f,g,psi,ham] = m3nObj(w, examples, lambda, decodeFunc)
%
% Outputs the objective value and gradient of the M3N learning objective
%
% w : current weights
% examples : array of example structure, each containing:
%	ss : nParam x 1 vector of sufficient statistics
%	Ynode : nState x nNode overcomplete matrix representation of labels
% lambda : regularization constant or vector
% decodeFunc : decoder function

% Initialization
nEx = length(examples);
psi = zeros(size(w));
ham = 0;

% Loop over examples
for i = 1:nEx

	ex = examples(i);
	nNodes = double(ex.edgeStruct.nNodes);
	
	% Loss-augmented inference
	[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,ex.edgeStruct);
	nodePot = nodePot .* exp(1 - ex.Ynode');
	yhat = decodeFunc(nodePot,edgePot,ex.edgeStruct);

	% Compute sufficient statistics
	ss_mu = sufficientStats(yhat,ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,ex.edgeStruct);

	% Hamming distance and difference of suff stats
	ham = ham + nnz(ex.Y ~= yhat) / (nEx * nNodes);
	psi = psi + (ss_mu - ex.ss) / (nEx * nNodes);

end

% Objective
f = 0.5*w'*(lambda.*w) + w'*psi + ham;

% Gradient
g = lambda.*w + psi;
