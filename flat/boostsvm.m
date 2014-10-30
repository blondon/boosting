function [models,alphas] = boostsvm(X, Y, T, liblinopt)
%
% Trains an ensemble of MRFs using max-margin BCFW.
%
% X : matrix of observations
% Y : vector of labels
% T : number of rounds of boosting
% liblinopt : optional string of libSVM arguments (includes reg constant C)

nEx = length(Y);

W = ones(nEx,1); % unnormalized distribution
dist = W / nEx;

models = {};
alphas = [];
err = [];

for t = 1:T
	% Train a model using current distribution
	models{t} = train(W,Y,X,liblinopt);
	% Compute error
	pred = predict(Y,X,models{t},'-q');
	errs = (Y ~= pred);
	err(t) = errs' * dist;
	if err > .5
		fprintf('Weak learning not satisfied; err_%d = %.3f. Aborting boosting.\n',t,err(t));
		break;
	end
	% Compute ensemble weight
	alphas(t) = 0.5 * log((1-err(t)) / (err(t)));
	% Update the distribution
	W = W .* exp(-alphas(t) * (2*errs - 1));
	dist = W ./ sum(W);
	
	fprintf('Round %d: err = %.3f, alpha_t = %.3f \n',t,err(t),alphas(t));
end

end


