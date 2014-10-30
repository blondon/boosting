
assert(exist('folds','var')==1,'flat_experiment requires variable ''folds'' to exist in the workspace.');

rng(311);

% SVM:1, AdaBoost:2
algoNames = {'SVM','AdaBoost'};
if ~exist('algos','var')
	algos = [1 2];
end
Cvec = 1;%[.01 .05 .1 .5 1 5 10];

% Number of rounds of boosting
T = 20;

% LIBLINEAR type
liblintype = 4;

% Error stats
statCols = {'Train Err','Test Err','Test Prec','Test Rec','Test F1'};
stats = inf(length(algoNames),length(folds),length(statCols));

% Model parameters
params = cell(length(algoNames),length(folds));

% Total number of jobs to complete
nJobs = length(folds) * (length(Cvec)*(length(Cvec)>1) + length(algos));
nJobsCompleted = 0;
totalTimer = tic;

nFolds = length(folds);
for f = 1:nFolds

	% Split data
	idx = circshift((1:nFolds)',f);
	trExamples = [folds{idx(1:nFolds-2)}];
	vaExamples = folds{idx(nFolds-1)};
	trExamplesFull = [trExamples vaExamples];
	teExamples = folds{idx(nFolds)};
	[Xtr,Ytr] = struct2flat(trExamples);
	[Xva,Yva] = struct2flat(vaExamples);
	[Xte,Yte] = struct2flat(teExamples);
	Xfull = [Xtr ; Xva]; Yfull = [Ytr ; Yva];
	
	% Use validation set to determine optimal lambda
	if length(Cvec) > 1 && any(algos == 1)
		vaErrs = zeros(length(Cvec),1);
		for l = 1:length(Cvec)
			C = Cvec(l);
			fprintf('---------------\nFold %d validation: C=%.4f \n',f,C);

			% SVM
			%libsvmopt = sprintf('-q -t 0 -c %f',C);
			%model = svmtrain([],Ytr,Xtr,libsvmopt);
			liblinopt = sprintf('-q -s %d -c %f',liblintype,C);
			model = train([],Ytr,Xtr,liblinopt);

			% Compute validation error
			vaErrs(l) = computeflatstats(model,Xva,Yva,trExamples,1);
			fprintf('Validation err: %.5f \n',vaErrs(l));

			% Update ETA
			nJobsCompleted = nJobsCompleted + 1;
			remainingtime(nJobs,nJobsCompleted,totalTimer);

		end
		l = find(vaErrs == min(vaErrs),1,'last');
		C = Cvec(l);
	else
		C = Cvec(1);
	end
	
	for a = algos

		fprintf('---------------\nFold %d (full): %s: C=%.4f \n',f,algoNames{a},C);
		model = [];
		
		if a == 1
			% SVM
			%libsvmopt = sprintf('-q -t 0 -c %f',C);
			%model = svmtrain([],Yfull,Xfull,libsvmopt);
			liblinopt = sprintf('-q -s %d -c %f',liblintype,C);
			model = train([],Yfull,Xfull,liblinopt);

		elseif a == 2
			% AdaBoost SVM
			liblinopt = sprintf('-q -s %d -c %f',liblintype,C);
			[model.models,model.alphas] = boostsvm(Xfull,Yfull,T,liblinopt);
			
		else
			die('no such algo');
		end

		% Train error
		[~,stats(a,f,1)] = computeflatstats(model,Xfull,Yfull,trExamplesFull,a);

		% Test error
		[~,stats(a,f,2)] = computeflatstats(model,Xte,Yte,teExamples,a);

		% Display the results of the current job in a nice tabular format
		disptable(squeeze(stats(a,f,1:2))',{'Train','Test'},algoNames{a},'%.5f');

		% Update ETA
		nJobsCompleted = nJobsCompleted + 1;
		remainingtime(nJobs,nJobsCompleted,totalTimer);

	end
	
end

% Average stats
avgStats = squeeze(mean(stats,2));
stdStats = squeeze(std(stats,[],2));

fprintf('Avg results (%d folds)\n',nFolds);
disptable(avgStats(algos,:),statCols,algoNames(algos),'%.5f');

sigthresh = .05;
fprintf('T-test (thresh=%.3f): %d \n',sigthresh,ttest(stats(1,:,2),stats(2,:,2),sigthresh));

