function [err,werr] = computeflatstats(model,X,Y,structExamples,algo)

if algo == 1
	preds = predict(Y,X,model);
elseif algo == 2
	nLabels = model.models{1}.nr_class;
	enspreds = zeros(length(Y),length(model.models));
	for t = 1:length(model.models)
		enspreds(:,t) = predict(Y,X,model.models{t});
	end
	preds = zeros(size(Y));
	for i = 1:length(Y)
		scores = zeros(nLabels,1);
		for t = 1:length(model.models)
			scores(enspreds(i,t)) = scores(enspreds(i,t)) + model.alphas(t);
		end
		[~,preds(i)] = max(scores);
	end
else
	die('no such algo');
end
err = nnz(Y ~= preds) / length(Y);
errs = zeros(length(structExamples),1);
offset = 0;
for i = 1:length(structExamples)
	ex = structExamples(i);
	nNodes = length(ex.Y);
	errs(i) = nnz(ex.Y ~= preds(offset+1:offset+nNodes)) / nNodes;
	offset = offset + nNodes;
end
werr = mean(errs);
