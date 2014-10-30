function folds = createFolds(examples,nFolds)

nPerFold = length(examples) / nFolds;

for f = 1:nFolds
	idx = ((f-1)*nPerFold+1):(f*nPerFold);
	folds{f} = examples(idx);
end
