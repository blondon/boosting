function [model,dual] = kernelfw(examples,decodeFunc,kernelPotFunc,kernelDotProdFunc,lambda,options)
%
% Trains an MRF using kernelized block coordinate frank-wolfe.
%
% examples : array of examples structs, each containing:
%	ss : nParam x 1 vector of sufficient statistics
%	Ynode : nState x nNode overcomplete matrix representation of labels
% decodeFunc : decoder function
% kernelPotFunc : kernel potentials function
% kernelDotProdFunc : kernel dot product function to compute (w' * w)
% lambda : optional regularization constant or vector (def: 1)
% options : optional struct of optimization options for subgradient descent:
% 			maxIter : iterations (def: 100*length(examples))
%			convCheck : check convergence every convCheck iterations
% 			tolerance : convergence tolerance (def: 1e-6)
% 			avgwindow : averaging window (def: maxIter)
% 			verbose : verbose mode (def: 0)
% 			plotObj : figure for objective plotting (def: 0, no plot)
% 			plotRefresh : number of iterations per plot refresh (def: 10)

% parse input
assert(nargin >= 4, 'USAGE: kernelfw(examples,decodeFunc,kernelPotFunc,kernelDotProdFunc)')
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

% Init variables
nEx = length(examples);
Z = lambda * nEx;
alphas = cell(length(examples),1);
avgalphas = cell(length(examples),1);
b = cell(length(examples),1);
for i = 1:nEx
	alphas{i} = containers.Map;
	avgalphas{i} = containers.Map;
	b{i} = containers.Map;
% 	ykey = kernelHash(examples(i).Y);
% 	alphas{i}(ykey) = {1,examples(i).Y};
% 	avgalphas{i}(ykey) = {1,examples(i).Y};
% 	b{i}(ykey) = 0;
end
lis = zeros(nEx,1);
[~,wtwmat] = kernelDotProd(examples,[],alphas,[],kernelDotProdFunc,Z);
% Averages
avgdual = 0;
avgloss = 0;
avgwtw = 0;

% Init plots
if options.plotObj
	set(0,'CurrentFigure',options.plotObj); clf;
	subplot(211); plot1Ax = gca;
	subplot(212); plot2Ax = gca;
	% History
	dualvec = 0;
	normwvec = 0;
	lossvec = 0;
end

for t = 1:options.maxIter
	
	% Pick a random example
	i = randi(nEx);
	ex = examples(i);
	
	% Loss-augmented inference
	[nodePot,edgePot] = kernelPots(ex,examples,alphas,kernelPotFunc,Z);
	yMAP = decodeFunc(nodePot.*exp(1-ex.Ynode'),edgePot,ex.edgeStruct);
	
	% Hash yMAP
	ykey = kernelHash(yMAP);
	
	% Hamming loss for worst violator
	ls = nnz(ex.Y ~= yMAP) / length(ex.Y);
	ls = ls / nEx;
	
	% Compute step size
	%   Simple version
	%gamma = 2*nEx / (2*nEx + t);
	%   Line search
	witwi = wtwmat(i,i);
	witw = sum(wtwmat(i,:));
	s = containers.Map; s(ykey) = {1,yMAP};
	wstws = kernelDotProd(examples(i),[],{s},[],kernelDotProdFunc,Z);
	[wstw,wstwvec] = kernelDotProd(examples(i),examples,{s},alphas,kernelDotProdFunc,Z);
	wstwi = wstwvec(i);
	gamma = (witw - wstw - (lis(i)-ls)/lambda) / (witwi - 2*wstwi + wstws);
	gamma = max(0, min(1, gamma));
	
	% Update maps
	if ~isKey(alphas{i},ykey)
		alphas{i}(ykey) = {0,yMAP};
		avgalphas{i}(ykey) = {0,yMAP};
		b{i}(ykey) = ls;
	end

	% Take steps and compute loss l_i = b' * alpha_i
	li = 0;
	for ykeys = alphas{i}.keys
		aikey = ykeys{:};
		alpha_iy = alphas{i}(aikey);
		if aikey == ykey
			alpha_iy{1} = (1-gamma) * alpha_iy{1} + gamma;
		else
			alpha_iy{1} = (1-gamma) * alpha_iy{1};
		end
		alphas{i}(aikey) = alpha_iy;
		li = li + b{i}(aikey) * alpha_iy{1};
	end
	lis(i) = li;
	
	% Loss
	loss = sum(lis);

	% Compute regularizer, w' * w = ||A * alpha||^2
	% Update row/column of wtwmat, then sum over elements of matrix
	[~,witwvec] = kernelDotProd(examples(i),examples,alphas(i),alphas,kernelDotProdFunc,Z);
	wtwmat(i,:) = witwvec; wtwmat(:,i) = witwvec;
	wtw = sum(wtwmat(:));
	
	% Dual objective
	dual = loss - 0.5 * lambda * wtw;
	
	% Averages
	k = min(t,options.avgwindow);
	for j = 1:nEx
		for ykeys = alphas{j}.keys
			ajkey = ykeys{:};
			alpha_jy = alphas{j}(ajkey);
			avgalpha_jy = avgalphas{j}(ajkey);
			avgalpha_jy{1} = k/(k+2) * avgalpha_jy{1} + 2/(k+2) * alpha_jy{1};
			avgalphas{j}(ajkey) = avgalpha_jy;
		end
	end
	avgwtw = k/(k+2) * avgwtw + 2/(k+2) * wtw;
	avgdual = k/(k+2) * avgdual + 2/(k+2) * dual;
	avgloss = k/(k+2) * avgloss + 2/(k+2) * loss;
	
	% Convergence criteria
	if mod(t,options.convCheck) == 0
		primal = 0.5 * lambda * wtw;
		[nodePots,edgePots] = kernelPotsBatch(examples,examples,alphas,kernelPotFunc,Z);
		for j = 1:nEx
			ex = examples(j);
			nodePot = nodePots{j};
			edgePot = edgePots{j};
			yMAP = decodeFunc(nodePot.*exp(1-ex.Ynode'),edgePot,ex.edgeStruct);
			hingeloss = ( ...
				computeEnergy(yMAP,nodePot,edgePot,ex.edgeStruct) ...
				- computeEnergy(ex.Y,nodePot,edgePot,ex.edgeStruct) ...
				+ nnz(ex.Y ~= yMAP)) / (nEx * length(ex.Y));
			primal = primal + hingeloss;
		end
		gap = primal - dual;
		if options.verbose
			fprintf('Primal: %f; Dual: %f; Gap: %f \n',primal,dual,gap);
		end
		converged = gap < options.tolerance;
	else
		converged = 0;
	end

	% Refresh plots
	if (options.plotObj && mod(t,options.plotRefresh) == 0)
		dualvec(end+1) = avgdual;
		normwvec(end+1) = avgwtw;
		lossvec(end+1) = avgloss;
		hAx = plotyy(plot1Ax,1:length(dualvec),dualvec,1:length(normwvec),normwvec);
		ylabel(hAx(1),'Dual'); ylabel(hAx(2),'||w||^2');
% 		xlabel(plot1Ax,sprintf('Iteration / %d',options.plotRefresh));
		plot(plot2Ax,1:length(lossvec),lossvec);
		ylabel(plot2Ax,'Loss');
		xlabel(plot2Ax,sprintf('Iteration / %d',options.plotRefresh));
		drawnow;
	end
	
	% Early stopping
	if converged
		break;
	end
	
end

% Always use average alphas because it has better convergence properties
alphas = avgalphas;

if t == options.maxIter
	if options.verbose && options.convCheck > 0
		fprintf('Frank-Wolfe did not converge in %d iterations. Using average alphas\n', t);
	end
	% Use average alphas because it has better convergence properties if we
	% never achieved certificate of optimality.
	%alphas = avgalphas;
else
	fprintf('Frank-Wolfe reached certificate of optimality within tolerance %s in %d iterations\n', ...
		options.tolerance, t);
end

% Fill model struct
model.examples = examples;
model.alphas = alphas;
model.kernelPotFunc = kernelPotFunc;
model.Z = Z;

