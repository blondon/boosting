function folds = loadCora(nFolds)

nPC = 100;
jumpRate = .1;

folds = {};
for f = 1:nFolds
	folds{f} = loadDocDataSnowball('cora.mat',3,nPC,jumpRate);
end
