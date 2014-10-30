function checkFx(w,y,Fx,Xnode,Xedge,nodeMap,edgeMap,edgeStruct)
%
% Verifies implementation of Fx and overcompleteRep.
%

nNodes = edgeStruct.nNodes;
nEdges = edgeStruct.nEdges;
maxStates = size(nodeMap,2);

% Guaranteed correct, but slow, version
[nodePot1,edgePot1] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct);
energy1 = computeEnergy(y,nodePot1,edgePot1,edgeStruct);

% Verify overcomplete rep
[~,~,ocrep] = overcompleteRep(y,edgeStruct);
nodePot1 = log(nodePot1); nodePot1(isinf(nodePot1)) = 0;
edgePot1 = log(edgePot1); edgePot1(isinf(edgePot1)) = 0;
wFx = [reshape(nodePot1',[],1) ; edgePot1(:)]';
energy2 = wFx * ocrep;
delta = abs(energy1-energy2);
if delta < 1e-10
	fprintf('Overcomplete rep test passed!\n');
else
	fprintf('Overcomplete rep test failed; delta=%f\n', delta);
end

% Fast, matrix-multiplication version
wFx = w' * Fx;
localScope = nNodes * maxStates;
nodePot2 = reshape(wFx(1:localScope),maxStates,nNodes)';
edgePot2 = reshape(wFx(localScope+1:end),maxStates,maxStates,nEdges);
ss1 = sufficientStats(y,Xnode,Xedge,nodeMap,edgeMap,edgeStruct);
ss2 = Fx * ocrep;
energy2 = wFx * ocrep;
delta = abs(energy1-energy2);
if delta < 1e-10
	fprintf('Fx test passed!\n');
else
	fprintf('Fx test failed; delta=%f\n', delta);
	delta = sum(abs(nodePot1(:)-nodePot2(:)));
	if delta < 1e-10
		fprintf('  nodePot test passed!\n');
	else
		fprintf('  nodePot test failed; delta=%f\n', delta);
	end
	delta = sum(abs(edgePot1(:)-edgePot2(:)));
	if delta < 1e-10
		fprintf('  edgePot test passed!\n');
	else
		fprintf('  edgePot test failed; delta=%f\n', delta);
	end
	delta = sum(abs(ss1-ss2));
	if delta < 1e-10
		fprintf('  suff. stats test passed!\n');
	else
		fprintf('  suff. stats test failed; delta=%f\n', delta);
	end
end

