function [nodePot,edgePot] = kernelPots(example, trainset, alphas, mexKernelFunc, Z)

% Init potentials for testex
maxStates = max(example.edgeStruct.nStates);
nodePot = zeros(length(example.Y),maxStates);
edgePot = zeros(maxStates,maxStates,example.edgeStruct.nEdges);

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
	% Compute kernelized potentials in mex function
	mexKernelFunc(int32(trainex.Y),int32(violators), ...
				nodePot,edgePot, ...
				trainex.edgeStruct.nStates,example.edgeStruct.nStates, ...
				trainex.edgeStruct.edgeEnds,example.edgeStruct.edgeEnds, ...
				trainex.Xnode,example.Xnode,trainex.Xedge,example.Xedge, ...
				alphavec);
end

% Normalize by number of active examples
nodePot = nodePot / Z;
edgePot = edgePot / Z;

% Exponentiate log-potentials
for n = 1:length(example.Y)
	nodePot(n,1:example.edgeStruct.nStates(n)) = exp(nodePot(n,1:example.edgeStruct.nStates(n)));
end
for e = 1:example.edgeStruct.nEdges
	n1 = example.edgeStruct.edgeEnds(e,1);
	n2 = example.edgeStruct.edgeEnds(e,2);
	edgePot(1:example.edgeStruct.nStates(n1),1:example.edgeStruct.nStates(n2),e) = ...
		exp(edgePot(1:example.edgeStruct.nStates(n1),1:example.edgeStruct.nStates(n2),e));
end

