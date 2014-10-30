/**
 * Some common kernel functions
 */

double linearKernel(double *x1, double *x2, int nFeats, int c1, int c2)
{
	int f;
	double k = 0.0;
	for (f = 0; f < nFeats; f++)
	{
		k += x1[f+nFeats*c1] * x2[f+nFeats*c2];
	}
	return k;
}

double rbfKernel(double sigma, double *x1, double *x2, int nFeats, int c1, int c2)
{
	int f;
	double diff;
	double k = 0.0;
	for (f = 0; f < nFeats; f++)
	{
		diff = x1[f+nFeats*c1] - x2[f+nFeats*c2];
		k += diff * diff;
	}
	return exp(-0.5 * k / sigma);
}
