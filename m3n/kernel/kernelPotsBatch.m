function [nodePots,edgePots] = kernelPotsBatch(examples, trainset, alphas, mexKernelFunc, Z)

% Init cell array of potentials
nodePots = cell(length(examples),1);
edgePots = cell(length(examples),1);
for j = 1:length(examples)
	ex = examples(j);
	maxStates = max(ex.edgeStruct.nStates);
	nodePots{j} = zeros(length(ex.Y),maxStates);
	edgePots{j} = zeros(maxStates,maxStates,ex.edgeStruct.nEdges);
end

% For each active example in alphas ...
activeExamples = find(~cellfun(@isempty,alphas));
for idx = 1:length(activeExamples)
	i = activeExamples(idx);
	% Get current example and alphas
	trainex = trainset(i);
	alpha_i = alphas{i};
	% Collect states,alphas for worst violators
	ykeys = alpha_i.keys;
	alphavec = zeros(length(ykeys),1);
	violators = zeros(length(trainex.Y),length(ykeys));
	for yi = 1:length(ykeys)
		alpha_iy = alpha_i(ykeys{yi});
		alphavec(yi) = alpha_iy{1};
		violators(:,yi) = alpha_iy{2};
	end
	for j = 1:length(examples)
		ex = examples(j);
		% Compute kernelized potentials in mex function
		mexKernelFunc(int32(trainex.Y),int32(violators), ...
					nodePots{j},edgePots{j}, ...
					trainex.edgeStruct.nStates,ex.edgeStruct.nStates, ...
					trainex.edgeStruct.edgeEnds,ex.edgeStruct.edgeEnds, ...
					trainex.Xnode,ex.Xnode,trainex.Xedge,ex.Xedge, ...
					alphavec);
	end
end

for j = 1:length(examples)
	
	ex = examples(j);
	nodePot = nodePots{j}; edgePot = edgePots{j};
	
	% Normalize by number of active examples
	nodePot = nodePot / Z;
	edgePot = edgePot / Z;

	% Exponentiate log-potentials
	for n = 1:length(ex.Y)
		nodePot(n,1:ex.edgeStruct.nStates(n)) = exp(nodePot(n,1:ex.edgeStruct.nStates(n)));
	end
	for e = 1:ex.edgeStruct.nEdges
		n1 = ex.edgeStruct.edgeEnds(e,1);
		n2 = ex.edgeStruct.edgeEnds(e,2);
		edgePot(1:ex.edgeStruct.nStates(n1),1:ex.edgeStruct.nStates(n2),e) = ...
			exp(edgePot(1:ex.edgeStruct.nStates(n1),1:ex.edgeStruct.nStates(n2),e));
	end
	
	nodePots{j} = nodePot; edgePots{j} = edgePot;
	
end

