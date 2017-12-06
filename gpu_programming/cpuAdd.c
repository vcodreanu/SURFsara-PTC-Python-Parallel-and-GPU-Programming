#include <stdio.h>

int main(void)
{
int N = 10;
float a[N],b[N],c[N];

for (int i = 0; i < N; ++i){
	a[i] = i;
	b[i] = 2.0f;	
}

for (int i = 0; i < N; ++i){
	c[i]= a[i]+b[i];	
}

for (int i = 0; i < N; ++i){
	printf("%f \n",c[i]);	
}


return 0;
}
