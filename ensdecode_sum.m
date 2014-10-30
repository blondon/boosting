function labels = ensdecode_sum(models,alphas,scoreFunc,Xnode,Xedge,nodeMap,edgeMap,edgeStruct)

nodeBel = zeros(edgeStruct.nNodes,edgeStruct.nStates(1));
for t = 1:length(models)
	[nodePot,edgePot] = UGM_CRF_makePotentials(...
		models(t).w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct);
	nodeBel = nodeBel + alphas(t) * scoreFunc(nodePot,edgePot,edgeStruct);
end

[~,labels] = max(nodeBel,[],2);
