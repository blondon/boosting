function [Ynode,Yedge,ocrep] = overcompleteRep(y, edgeStruct)

% Converts a vector of values to overcomplete representation.
%
% vals : nNode x 1 vector of state values
% nState : number of states per variable (must be uniform)
%
% Ynode : nState x nNode overcomplete representation
% Yedge : nState x nState x nNode overcomplete representation
% ocrep : nState*nNode x 1 overcomplete representation

maxStates = max(edgeStruct.nStates(:));

Ynode = zeros(maxStates,edgeStruct.nNodes);

for n = 1:edgeStruct.nNodes
	Ynode(y(n),n) = 1;
end

if nargout >= 2
	Yedge = zeros(maxStates,maxStates,edgeStruct.nEdges);
	for e = 1:edgeStruct.nEdges
		n1 = edgeStruct.edgeEnds(e,1);
		n2 = edgeStruct.edgeEnds(e,2);
		Yedge(y(n1),y(n2),e) = 1;
	end
end

if nargout >= 3
	ocrep = [Ynode(:) ; Yedge(:)];
end
