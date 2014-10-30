function [w,primal] = bcfw(examples, decodeFunc, lambda, options, w)
%
% Trains an MRF using max-margin block coordinate frank-wolfe.
%
% examples : array of examples structs, each containing:
%	ss : (nParam x 1) vector of sufficient statistics
%	Ynode : (nState x nNode) overcomplete matrix representation of labels
% decodeFunc : decoder function
% lambda : optional regularization constant or vector (def: 1)
% options : optional struct of optimization options for subgradient descent:
% 			maxIter : iterations (def: 100*length(examples))
%			convCheck : check convergence every convCheck iterations
% 			tolerance : convergence tolerance (def: 1e-6)
% 			avgwindow : averaging window (def: maxIter)
% 			verbose : verbose mode (def: 0)
% 			plotObj : figure for objective plotting (def: 0, no plot)
% 			plotRefresh : number of iterations per plot refresh (def: 10)
%			dist : (nEx x 1) distribution over examples (def: uniform)
% w : init weights (optional: def=zeros)

% parse input
assert(nargin >= 2, 'USAGE: bcfw(examples,decodeFunc)')
if ~exist('lambda','var') || isempty(lambda)
	lambda = 1;
end
if ~exist('options','var') || ~isstruct(options)
	options = struct();
end
if ~isfield(options,'maxIter')
	options.maxIter = 100 * length(examples);
end
if ~isfield(options,'convCheck')
	options.convCheck = 0;
end
if ~isfield(options,'tolerance')
	options.tolerance = 1e-6;
end
if ~isfield(options,'avgwindow')
	options.avgwindow = options.maxIter;
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
if isfield(options,'dist')
	dist = options.dist;
	primalObj = @(w) m3nObj_weighted(w,examples,lambda,decodeFunc,dist);
else
	dist = cell(length(examples),1);
	primalObj = @(w) m3nObj(w,examples,lambda,decodeFunc);
end
if ~exist('w','var') || isempty(w)
	nParam = max(examples(1).edgeMap(:));
	w = zeros(nParam,1);
end

% Init variables
nEx = length(examples);
wi = zeros(length(w),nEx);
li = zeros(nEx,1);
l = 0;
primalavg = 0;
dualavg = 0;
gapavg = 0;
wavg = w;

% Init plots
if options.plotObj
	set(0,'CurrentFigure',options.plotObj);
	clf;
	% primal objective, norm(x) plot
	subplot(211);
	objAx = gca;
	% dual objective, gap plot
	subplot(212);
	dualAx = gca;
	% History
	primalvec = primalObj(w);
	normwvec = w' * w;
	dualvec = 0;
	gapvec = primalvec(1) - dualvec(1);
end

% Create a random sequence of examples
example_seq = randsample(nEx,options.maxIter,true);
% if isfield(options,'dist')
% 	example_seq = randsample(nEx,options.maxIter,true,options.dist);
% else
% 	example_seq = randsample(nEx,options.maxIter,true);
% end

for t = 1:options.maxIter
	
	% Get the next example in the sequence
	i = example_seq(t);
	ex = examples(i);
	nNodes = double(ex.edgeStruct.nNodes);
	
	% Compute primal, difference of sufficient stats and loss
	% Note: this is a hack at estimating the primal based on a single point.
	if isempty(dist{i})
		[primal,~,psi,ls] = m3nObj(w,ex,lambda,decodeFunc);
		% m3nObj divides psi and ls by number of input examples,
		% which is 1 when using stochastic updates.
		% FW needs these divided by the actual nEx in the dataset.
		psi = psi ./ nEx; ls = ls / nEx;
	else
		% Normalize the distribution by the mass on the current example.
		d = dist{i} / sum(dist{i});
		[primal,~,psi,ls] = m3nObj_weighted(w,ex,lambda,decodeFunc,{d});
	end
	
	% Gradient of w at coordinate s
	ws = -psi ./ lambda;

	% Compute step size
	%   Simple version
	%gamma = 2*nEx / (2*nEx + t);
	%   Line search
	wimws = wi(:,i) - ws;
	gamma = (w'*(lambda.*wimws) - li(i) + ls) / (wimws'*(lambda.*wimws));
	gamma = max(0, min(1, gamma));

	% Take step(s)
	wiPrev = wi(:,i);
	liPrev = li(i);
	wi(:,i) = (1-gamma) * wiPrev + gamma * ws;
	li(i) = (1-gamma) * liPrev + gamma * ls;
	w = w + wi(:,i) - wiPrev;
	l = l + li(i) - liPrev;
	
	% Duality gap
	dual = l - 0.5 * w' * (lambda .* w);
	gap = primal - dual;
	
	% Averages
	k = min(t,options.avgwindow);
	wavg = k/(k+2) * wavg + 2/(k+2) * w;
	primalavg = k/(k+2) * primalavg + 2/(k+2) * primal;
	dualavg = k/(k+2) * dualavg + 2/(k+2) * dual;
	gapavg = k/(k+2) * gapavg + 2/(k+2) * gap;
	
	% Convergence criteria
	if nEx == 1
		converged = gap < options.tolerance;
	elseif mod(t,options.convCheck) == 0
		primal = primalObj(w);
		dual = sum(li) - 0.5 * w' * (lambda .* w);
		gap = primal - dual;
		if options.verbose
			fprintf('Primal: %f; Dual: %f; Gap: %f \n',primal,dual,gap);
		end
		converged = gap < options.tolerance;
	else
		converged = 0;
	end
	
	% Refresh plots
	if (options.plotObj && mod(t,options.plotRefresh) == 0) || converged
		% Update history
		primalvec(end+1) = primalavg;
		normwvec(end+1) = wavg' * wavg;
		dualvec(end+1) = dualavg;
		gapvec(end+1) = gapavg;
		set(0,'CurrentFigure',options.plotObj);
		% Primal objective and norm(w)
		hAx = plotyy(1:length(primalvec),primalvec,1:length(normwvec),normwvec,'Parent',objAx);
		ylabel(hAx(1),'Avg. Primal Obj.'); ylabel(hAx(2),'||w_{avg}||^2');
		% Dual objective and duality gap
		hAx = plotyy(1:length(dualvec),dualvec,1:length(dualvec),gapvec,'Parent',dualAx);
		ylabel(hAx(1),'Avg. Dual Obj.'); ylabel(hAx(2),'Avg. Duality Gap');
		xlabel(dualAx,sprintf('Iteration / %d',options.plotRefresh));
		drawnow;
	end
	
	% Early stopping
	if converged
		break;
	end
	
end

if nargout == 2
	primal = primalObj(w);
	dual = sum(li) - 0.5 * w' * (lambda .* w);
	gap = primal - dual;
	if options.verbose
		fprintf('Primal: %f; Dual: %f; Gap: %f \n',primal,dual,gap);
	end
end

if t == options.maxIter
	if options.verbose && (nEx == 1 || options.convCheck > 0)
		fprintf('Frank-Wolfe did not converge in %d iterations.\n', t);
	end
	% Use average w because it has better convergence properties if we
	% never achieved a certificate of convergence
	w = wavg;
else
	fprintf('Frank-Wolfe reached certificate of optimality within tolerance %f in %d iterations.\n', ...
		options.tolerance, t);
end

