function ex = makeExample(Xnode, Xedge, y, edgeStruct)
%
% Makes an example structure.
%

[nodeMap,edgeMap] = UGM_makeCRFmaps(Xnode,Xedge,edgeStruct,0,1,1);
edgeStruct.nNodeFeats = size(Xnode,2) * ones(edgeStruct.nNodes,1);
edgeStruct.nEdgeFeats = size(Xedge,2) * ones(edgeStruct.nEdges,1);
edgeStruct.nParams = max(edgeMap(:));
edgeStruct.nNodeParams = max(nodeMap(:));
edgeStruct.nEdgeParams = edgeStruct.nParams - edgeStruct.nNodeParams;
ex.edgeStruct = edgeStruct;
ex.nodeMap = nodeMap;
ex.edgeMap = edgeMap;
ex.Y = int32(y);
ex.Ynode = overcompleteRep(y,edgeStruct);
ex.Xnode = Xnode;
ex.Xedge = Xedge;
ex.ss = sufficientStats(y,Xnode,Xedge,nodeMap,edgeMap,edgeStruct);
