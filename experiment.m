
assert(exist('folds','var')==1,'experiment requires variable ''folds'' to exist in the workspace.');

rng(311);

% M3N:1, BM3N:2
algoNames = {'M3N','BM3N'};
if ~exist('algos','var')
	algos = [1 2];
end
lambdas = [.0001 .0005 .001 .005 .01];

% Max-margin optimization params
maxIters = 1e3;%1e4;
plotRefresh = maxIters / 10;

% Ensemble scoring function
scoreFunc = @UGM_Infer_MaxMarginals;

% M3N Decoder
decodeFunc = @(nodePot,edgePot,edgeStruct) ...
	UGM_Decode_MaxOfMarginals(nodePot,edgePot,edgeStruct,scoreFunc);

% Number of rounds of boosting
T = 10;

% Number of folds
nFolds = length(folds);

% Error stats
statCols = {'Error','Prec','Rec','F1'};
trStats = inf(length(algoNames),nFolds,length(statCols));
teStats = inf(length(algoNames),nFolds,length(statCols));

% Model parameters
params = cell(length(algoNames),nFolds);

% Objective plots
fig = figure(101);

% Total number of jobs to complete
nJobs = nFolds * (...
	(length(lambdas)>1)*length(lambdas) + ...
	any(algos==1) + any(algos==2)*T);
nJobsCompleted = 0;
totalTimer = tic;

for f = 1:nFolds

	% Split data
	idx = circshift((1:length(folds))',f);
	trExamples = [folds{idx(1:length(folds)-2)}];
	vaExamples = folds{idx(length(folds)-1)};
	trExamplesFull = [trExamples vaExamples];
	teExamples = folds{idx(length(folds))};
	
	% Use validation set to determine optimal lambda
	if length(lambdas) > 1
		vaErrs = zeros(length(lambdas),1);
		for l = 1:length(lambdas)
			lam = lambdas(l);
			fprintf('---------------\nFold %d validation: lam=%.4f \n',f,lam);
			clf

			% M3N with BCFW
			options = struct('verbose',0,'plotObj',fig,'plotRefresh',plotRefresh, ...
							 'convCheck',0,'maxIter',maxIters);
			model = bcfw(trExamples,decodeFunc,lam,options);

			% Compute validation error
			vaErrs(l) = computestats(model,vaExamples,scoreFunc);
			fprintf('Validation err: %.5f \n',vaErrs(l));

			% Update ETA
			nJobsCompleted = nJobsCompleted + 1;
			remainingtime(nJobs,nJobsCompleted,totalTimer);

		end
		l = find(vaErrs == min(vaErrs),1,'last');
		lam = lambdas(l);
	else
		lam = lambdas(1);
	end
	
	for a = algos

		fprintf('---------------\nFold %d (full): %s: lam=%.4f \n',f,algoNames{a},lam);
		clf
		model = [];
		
		switch a
			case 1
				% M3N with BCFW
				options = struct('verbose',0,'plotObj',fig,'plotRefresh',plotRefresh, ...
								 'convCheck',0,'maxIter',maxIters);
				model = bcfw(trExamplesFull,decodeFunc,lam,options);
				nJobsCompleted = nJobsCompleted + 1;

			case 2
				options = struct('verbose',0,'plotObj',fig,'plotRefresh',plotRefresh, ...
								 'convCheck',0,'maxIter',maxIters);
				[model.models,model.alphas] = boostm3n(trExamplesFull,T,scoreFunc,lam,options);
				nJobsCompleted = nJobsCompleted + T;

		end

		% Train error
		[trStats(a,f,1)] = computestats(model,trExamplesFull,scoreFunc);

		% Test error
		[teStats(a,f,1)] = computestats(model,teExamples,scoreFunc);

		% Display the results of the current job in a nice tabular format
		disptable(squeeze([trStats(a,f,:);teStats(a,f,:)]),statCols,{'Train','Test'},'%.5f');

		% Update ETA
		remainingtime(nJobs,nJobsCompleted,totalTimer);

	end
	
end

% Average over all (or certain) folds
avgTrainResults = squeeze(mean(trStats,2));
stdTrainResults = squeeze(std(trStats,[],2));
avgTestResults = squeeze(mean(teStats,2));
stdTestResults = squeeze(std(teStats,[],2));

fprintf('Avg results (%d folds)\n',nFolds);
table = [avgTrainResults(algos,1) stdTrainResults(algos,1) avgTestResults(algos,1) stdTestResults(algos,1)];
tableCols = {'Train err','std','Test err','std'};
disptable(table,tableCols,algoNames(algos),'%.5f');

sigthresh = .05;
fprintf('T-test (thresh=%.3f): %d \n',sigthresh,ttest(teStats(1,:,1),teStats(2,:,1),sigthresh));

