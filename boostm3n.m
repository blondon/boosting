function [models,alphas,avgmargins] = boostm3n(examples, T, scoreFunc, lambda, options)
%
% Trains an ensemble of MRFs using max-margin BCFW.
%
% examples : array of examples structs, each containing:
%	ss : nParam x 1 vector of sufficient statistics
%	Ynode : nState x nNode overcomplete matrix representation of labels
% T : number of rounds of boosting
% scoreFunc : scoring function with signature:
%				nodeBel = scoreFunc(nodePot,edgePot,edgeStruct)
% lambda : optional regularization constant or vector (def: 1)
% options : optional struct of optimization options for subgradient descent:
% 			maxIter : iterations (def: 100*length(examples))
%			convCheck : check convergence every convCheck iterations
% 			tolerance : convergence tolerance (def: 1e-6)
% 			avgwindow : averaging window (def: maxIter)
% 			verbose : verbose mode (def: 0)
% 			plotObj : figure for objective plotting (def: 0, no plot)
% 			plotRefresh : number of iterations per plot refresh (def: 10)

nEx = length(examples);

dist = cell(nEx,1);
for i = 1:nEx
	nNodes = length(examples(i).Y);
	dist{i} = ones(nNodes,1) / (nEx * nNodes);
end
% errs = zeros(nEx,1);
margins = cell(nEx,1);

avgmargins = [];
avgerrs = [];
models = [];
alphas = [];

decodeFunc = @(nodePot,edgePot,edgeStruct) ...
	UGM_Decode_MaxOfMarginals(nodePot,edgePot,edgeStruct,scoreFunc);

for t = 1:T
	% Train a model
	options.dist = dist;
	w = bcfw(examples,decodeFunc,lambda,options);
	% Compute margins
	avgmargins(t) = 0;
	avgerrs(t) = 0;
	for i = 1:nEx
		ex = examples(i);
		[nodePot,edgePot] = UGM_CRF_makePotentials(...
			w,ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,ex.edgeStruct);
		nodeBel = scoreFunc(nodePot,edgePot,ex.edgeStruct);
		margins{i} = decompmargins(nodeBel,ex.Ynode');
		avgmargins(t) = avgmargins(t) + dist{i}' * margins{i};
		[~,pred] = max(nodeBel,[],2);
		avgerrs(t) = avgerrs(t) + dist{i}' * (pred ~= ex.Y);
	end
	% Check weak learning condition
	if avgmargins(t) < 0
		fprintf('Weak learning not satisfied; avg_margin_%d = %.3f. Aborting boosting.\n',t,avgmargins(t));
		% If this is the first round of boosting, just return the model
		% with weight 1.
		if t == 1
			models(t).w = w;
			alphas(t) = 1;
		end
		break
	end
	% Weak learning satisfied; use model
	models(t).w = w;
	% Compute ensemble weight
	alphas(t) = 0.5 * log((1+avgmargins(t)) / (1-avgmargins(t)));
	% Update the distribution
	Z = 0;
	for i = 1:nEx
		dist{i} = dist{i} .* exp(-alphas(t) * margins{i});
		Z = Z + sum(dist{i});
	end
	for i = 1:nEx
		dist{i} = dist{i} ./ Z;
	end
	
	fprintf('Round %d: avg_margin = %.3f, avg_err = %.3f, alpha_t = %.3f \n',t,avgmargins(t),avgerrs(t),alphas(t));
end

end


% Decomposed margins
function margins = decompmargins(nodeBel,Ynode)

score_y = nodeBel(Ynode==1);
score_max = max((~Ynode).*nodeBel,[],2);
margins = (score_y - score_max);

end

