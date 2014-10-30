function Fx = makeFx(Xnode, Xedge, nodeMap, edgeMap, edgeStruct)
%
% Creates the feature map for M3N learning.
%
% Xnode : 1 x nNodeFeat x nNode matrix of observed node features.
% Xedge : 1 x nEdgeFeat x nEdge matrix of observed edge features.
% nodeMap : UGM parameter map for nodes
% edgeMap : UGM parameter map for edges
% edgeStruct : UGM edge struct

% Dimensions
nNodes = double(edgeStruct.nNodes);
nEdges = double(edgeStruct.nEdges);
nStates = double(edgeStruct.nStates);
maxStates = max(nStates);
nNodeFeats = edgeStruct.nNodeFeats;
nEdgeFeats = edgeStruct.nEdgeFeats;
nParamLoc = double(max(nodeMap(:)));
nParamRel = double(max(edgeMap(:))) - nParamLoc;
nStateLoc = nNodes * maxStates;
nStateRel = nEdges * maxStates^2;

% Create local map
F_loc = zeros(nParamLoc,maxStates,nNodes);
for n = 1:nNodes
	for s = 1:nStates(n)
		for f = 1:nNodeFeats(n)
			p = double(nodeMap(n,s,f));
			F_loc(p,s,n) = Xnode(1,f,n);
		end
	end
end

% Create relational map
F_rel = zeros(nParamRel,maxStates,maxStates,nEdges);
for e = 1:nEdges
	n1 = edgeStruct.edgeEnds(e,1);
	n2 = edgeStruct.edgeEnds(e,2);
	for s1 = 1:nStates(n1)
		for s2 = 1:nStates(n2)
			for f = 1:nEdgeFeats(e)
				p = double(edgeMap(s1,s2,e,f)) - nParamLoc;
				F_rel(p,s1,s2,e) = Xedge(1,f,e);
			end
		end
	end
end

% Combine maps
%  Sparse method (faster)
Fx = [sparse(F_loc(:,:)) sparse(nParamLoc,nStateRel) ; sparse(nParamRel,nStateLoc) sparse(F_rel(:,:))];
%  Dense method
% Fx = zeros(nParamLoc+nParamRel,nStateLoc+nStateRel);
% Fx(1:nParamLoc,1:nStateLoc) = F_loc(:,:);
% Fx(nParamLoc+1:end,nStateLoc+1:end) = F_rel(:,:);
