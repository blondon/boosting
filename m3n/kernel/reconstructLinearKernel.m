function w = reconstructLinearKernel(nParams, examples, alphas, Z)

activeExamples = find(~cellfun(@isempty,alphas));
w = zeros(nParams,1);
for i = 1:length(activeExamples)
	ex = examples(activeExamples(i));
	alpha_i = alphas{activeExamples(i)};
	% For each "worst violator" assignment ...
	violators = alpha_i.keys;
	for j = 1:length(violators)
		% Get violator assignment
		alpha_iy = alpha_i(violators{j});
		y = alpha_iy{2};
		% Compute sufficient stats for configuration
		ss = sufficientStats(y,ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,ex.edgeStruct);
		w = w + alpha_iy{1} * (ex.ss-ss) / length(ex.Y);
	end
end
w = w / Z;
