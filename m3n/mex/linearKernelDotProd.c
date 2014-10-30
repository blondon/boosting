#include <math.h>
#include "mex.h"
#include "kernels.h"

/**
 * Computes the dot product for linear kernel StructSVM.
 *
 * INPUTS
 * V1,V2
 * nStates1,nStates2
 * edgeEnds1,edgeEnds2
 * Xnode1,Xnode2
 * Xedge1,Xedge2
 * alphas1,alphas2
 *
 * OUTPUTS
 * wtw = w1' * w2
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	/* Input variables */
	int *Y1, *Y2;
	int *V1, *V2;
	int *nStates1, *nStates2;
	int *edgeEnds1, *edgeEnds2;
	double *Xnode1, *Xnode2;
	double *Xedge1, *Xedge2;
	double *alphas1, *alphas2;
	
	/* Output variables */
	double wtw;
	double *outPtr;
	
	/* Local variables */
	int nNodes1, nNodes2;
	int nEdges1, nEdges2;
	int nNodeFeats;
	int nEdgeFeats;
	int nViolators1, nViolators2;
	int n1, n2, n1_1, n2_1, n1_2, n2_2;
	int e1, e2;
	int ys1, ys2, ys1_1, ys2_1, ys1_2, ys2_2;
	int vs1, vs2, vs1_1, vs2_1, vs1_2, vs2_2;
	int cnt;
	int v1, v2;
	int kernelComputed;
	double k;
	double Z;
	
	/* Input */
	Y1 = (int*)mxGetPr(prhs[0]);
	Y2 = (int*)mxGetPr(prhs[1]);
	V1 = (int*)mxGetPr(prhs[2]);
	V2 = (int*)mxGetPr(prhs[3]);
	nStates1 = (int*)mxGetPr(prhs[4]);
	nStates2 = (int*)mxGetPr(prhs[5]);
	edgeEnds1 = (int*)mxGetPr(prhs[6]);
	edgeEnds2 = (int*)mxGetPr(prhs[7]);
	Xnode1 = (double*)mxGetPr(prhs[8]);
	Xnode2 = (double*)mxGetPr(prhs[9]);
	Xedge1 = (double*)mxGetPr(prhs[10]);
	Xedge2 = (double*)mxGetPr(prhs[11]);
	alphas1 = (double*)mxGetPr(prhs[12]);
	alphas2 = (double*)mxGetPr(prhs[13]);
	
	/* Output */
	plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
	outPtr = (double*)mxGetPr(plhs[0]);
	
	/* Dimensions */
	nNodes1 = mxGetDimensions(prhs[0])[0];
	nNodes2 = mxGetDimensions(prhs[1])[0];
	nEdges1 = mxGetDimensions(prhs[6])[0];
	nEdges2 = mxGetDimensions(prhs[7])[0];
	nNodeFeats = mxGetDimensions(prhs[8])[1];
	nEdgeFeats = mxGetDimensions(prhs[10])[1];
	nViolators1 = mxGetDimensions(prhs[2])[1];
	nViolators2 = mxGetDimensions(prhs[3])[1];
	
	/* Normalize the kernel by (nNodes1 * nNodes2) */
	Z = ((double)nNodes1) * ((double)nNodes2);
	
	/* Init wtw */
	wtw = 0.0;
	
	/* Compute node energy */
	for (n1 = 0; n1 < nNodes1; n1++)
	{
		for (n2 = 0; n2 < nNodes2; n2++)
		{
			/* Reset kernel computation for every pair of nodes */
			kernelComputed = 0;
			/* Determine if nodes are of same type */
			if ((nStates1[n1] == nStates2[n2]))
			{
				ys1 = Y1[n1]-1;
				ys2 = Y2[n2]-1;
				/* Iterate over worst violators */
				for (v1 = 0; v1 < nViolators1; v1++)
				{
					vs1 = V1[n1+nNodes1*v1]-1;
					for (v2 = 0; v2 < nViolators2; v2++)
					{
						vs2 = V2[n2+nNodes2*v2]-1;
						/* Determine how many times to add alpha1*alpha2*k(x,x') */
						cnt = (vs1 == vs2) + (ys1 == ys2) - (vs1 == ys2) - (ys1 == vs2);
						if (cnt != 0)
						{
							/* Only compute kernel for these nodes if not already computed */
							if (kernelComputed == 0)
							{
								k = linearKernel(Xnode1,Xnode2,nNodeFeats,n1,n2) / Z;
								kernelComputed = 1;
							}
							/* Accumulate product */
							wtw += alphas1[v1] * alphas2[v2] * k * ((double)cnt);
						}
					}
				}
			}
		}
	}

	/* Compute edge energy */
	for (e1 = 0; e1 < nEdges1; e1++)
	{
		n1_1 = edgeEnds1[e1]-1;
		n2_1 = edgeEnds1[e1+nEdges1]-1;
		for (e2 = 0; e2 < nEdges2; e2++)
		{
			n1_2 = edgeEnds2[e2]-1;
			n2_2 = edgeEnds2[e2+nEdges2]-1;
			/* Reset kernel computation for every pair of edges */
			kernelComputed = 0;
			/* Determine if edges are of same type */
			if ((nStates1[n1_1] == nStates2[n1_2]) && (nStates1[n2_1] == nStates2[n2_2]))
			{
				ys1_1 = Y1[n1_1]-1;
				ys2_1 = Y1[n2_1]-1;
				ys1_2 = Y2[n1_2]-1;
				ys2_2 = Y2[n2_2]-1;
				/* Iterate over worst violators */
				for (v1 = 0; v1 < nViolators1; v1++)
				{
					vs1_1 = V1[n1_1+nNodes1*v1]-1;
					vs2_1 = V1[n2_1+nNodes1*v1]-1;
					for (v2 = 0; v2 < nViolators2; v2++)
					{
						vs1_2 = V2[n1_2+nNodes2*v2]-1;
						vs2_2 = V2[n2_2+nNodes2*v2]-1;
						/* Determine how many times to add alpha1*alpha2*k(x,x') */
						cnt = ((vs1_1 == vs1_2) && (vs2_1 == vs2_2))
							+ ((ys1_1 == ys1_2) && (ys2_1 == ys2_2))
							- ((vs1_1 == ys1_2) && (vs2_1 == ys2_2))
							- ((ys1_1 == vs1_2) && (ys2_1 == vs2_2));
						if (cnt != 0)
						{
							/* Only compute kernel for these nodes if not already computed */
							if (kernelComputed == 0)
							{
								k = linearKernel(Xedge1,Xedge2,nEdgeFeats,e1,e2) / Z;
								kernelComputed = 1;
							}
							/* Accumulate product */
							wtw += alphas1[v1] * alphas2[v2] * k * ((double)cnt);
						}
					}
				}
			}
		}
	}
	
	/* Assign to output */
	outPtr[0] = wtw;
}