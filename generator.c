#include<stdio.h>
#include<stdlib.h>
#include<complex.h>
#include<time.h>
#include<math.h>

int main(int argc, char** argv) {
	int n = atoi(argv[1]);
	double complex* v = (double complex*)malloc(n * sizeof(double complex));
	srand(time(NULL));
	for(int i = 0; i < n; ++i) {
		int re = rand() % 20;
		int im = rand() % 20;
		v[i] = re + im * I;
	}
	double norm = 0.0;
	for(int i = 0; i < n; ++i) {
		printf("%lf, %lf\n", creal(v[i]), cimag(v[i]));
		norm += creal(v[i]) * creal(v[i]) + cimag(v[i]) * cimag(v[i]);
	}
	norm = sqrt(norm);
	printf("%.14lf\n", norm);
	FILE* f = fopen("test1.dat", "wb");
	fwrite(v, sizeof(double complex), n, f);
	return 0;
}
