function [err,prec,rec,f1] = computestats(model,data,scoreFunc)

errs = zeros(length(data),1);

for i = 1:length(data)
	ex = data(i);
	if ~isstruct(model)
		[nodePot,edgePot] = UGM_CRF_makePotentials(model,...
			ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,ex.edgeStruct);
		pred = UGM_Decode_MaxOfMarginals(nodePot,edgePot,ex.edgeStruct,scoreFunc);
	else
		pred = ensdecode_sum(model.models,model.alphas,scoreFunc,...
			ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,ex.edgeStruct);
	end
	errs(i) = nnz(pred ~= ex.Y) / length(ex.Y);
end
err = mean(errs);


