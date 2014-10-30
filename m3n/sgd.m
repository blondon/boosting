function [w, favg, fvec] = sgd(examples, decodeFunc, lambda, options, w)
%
% Trains an MRF using max-margin formulation.
%
% examples : array of example structs, each containing:
%	suffStat : nParam x 1 vector of sufficient statistics
%	Ynode : nState x nNode overcomplete matrix representation of labels
% decodeFunc : decoder function
% lambda : optional regularization constant or vector (def: 1)
% options : optional struct of optimization options for subgradient descent:
% 			maxIter : iterations (def: 100*length(examples))
% 			stepSize : step size (def: 1)
% 			verbose : verbose mode (def: 0)
% w : init weights (optional: def=zeros)

% parse input
assert(nargin >= 2, 'USAGE: sgd(examples,decodeFunc)')
if ~exist('lambda','var') || isempty(lambda)
	lambda = 1;
end
if ~exist('options','var') || isempty(options)
	options = struct();
end
if ~isfield(options,'maxIter')
	options.maxIter = 100 * length(examples);
end
if ~isfield(options,'stepSize')
	options.stepSize = 1;
end
if ~isfield(options,'verbose')
	options.verbose = 0;
end
if ~isfield(options,'plotObj')
	options.plotObj = false;
end
if ~isfield(options,'plotRefresh')
	options.plotRefresh = 10;
end
if ~exist('w','var') || isempty(w)
	nParam = max(examples(1).edgeMap(:));
	w = zeros(nParam,1);
end

nEx = length(examples);

%% Main loop

favg = 0;
if nargout >= 3 || options.plotObj
	fvec = zeros(options.maxIter,1);
	wnrm = zeros(options.maxIter,1);
end

for t = 1:options.maxIter

	% Select a random example
	i = ceil(rand() * nEx);
	ex = examples(i);
	
	% Compute objective and gradient
	[f, g] = m3nObj(w, ex, lambda, decodeFunc);

	% Update estimate of function value
	favg = (1/t) * f + ((t-1)/t) * favg;

	% Update point
	w = w - (options.stepSize ./ t) .* g;

	% Store objective and weight norm
	if nargout >= 3 || options.plotObj
		fvec(t) = favg;
		wnrm(t) = w'*w;
	end

	% Console log
	if options.verbose
		fprintf('Iter = %d of %d (ex %d: f = %f, fAvg = %f)\n', t, options.maxIter, i, f, favg);
	end

	% Plot objective and weight norm
	if options.plotObj && mod(t,options.plotRefresh) == 0
		set(0,'CurrentFigure',options.plotObj);
		hAx = plotyy(1:t,fvec(1:t), 1:t,wnrm(1:t));
		ylabel(hAx(1),'Objective'); ylabel(hAx(2),'norm(w)^2');
		drawnow;
	end

end

% Final plot
if options.plotObj
	set(0,'CurrentFigure',options.plotObj);
	hAx = plotyy(1:length(fvec),fvec, 1:length(wnrm),wnrm);
	ylabel(hAx(1),'Objective'); ylabel(hAx(2),'norm(w)^2');
	drawnow;
end
