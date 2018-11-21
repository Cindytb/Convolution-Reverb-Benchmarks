#include <stdio.h> 
#include <string.h> 
#include <math.h> 

// A utility function to swap characters 
void swap (float *a, float *b){
	float temp = *a;
	*a = *b;
	*b = temp;
}
// A utility function to reverse string str[low..high] 
void reverse (float *arr, int low, int high){
	while( low < high){		
		swap (&arr[low], &arr[high]);
		low++;
		high--;
	}
}
// Cycle leader algorithm to move all even positioned elements 
// at the end. 
void cycleLeader( float*buf, int shift, int len){
	int j; 
	float item; 

	for (int i = 1; i < len; i *= 3 ) 
	{ 
		j = i; 

		item = buf[j + shift]; 
		do
		{ 
			// odd index 
			if ( j & 1 ) 
				j = len / 2 + j / 2; 
			// even index 
			else
				j /= 2; 

			// keep the back-up of element at new position 
			swap (&buf[j + shift], &item); 
		} 
		while ( j != i ); 
	} 
}
void printArr(float *buf, int len){
	for(int i = 0; i < len - 1; i++){
		printf("%f ", buf[i]);
	}
	printf("%f\n", buf[len - 1]);
}

void deInterleave(float *buf, long long samples){
	long long lenRemaining = samples, shift = 0;
	while(lenRemaining){
		int k = floor(log(lenRemaining - 1) / log(3));
		int currLen = pow(3, k) + 1;
		lenRemaining -= currLen;
		cycleLeader(buf, shift, currLen);
		reverse(buf, shift / 2, shift - 1);
		reverse(buf, shift, shift + currLen/2 - 1);
		reverse(buf, shift / 2, shift + currLen/2 - 1);
		shift += currLen;
	}
}
void interleave(float *buf, long long frames){
	long long start = frames, j = frames, i = 0;
	for(long long done = 0; done < 2 * frames - 2; done++){
		if (start == j){
			start--;
			j--;
		}
		
		i = j > frames - 1 ? j - frames: j;
		j = j > frames - 1 ? 2 * i + 1 : 2 * i;
		printf("start: %lli\t j: %lli\n", start, j);	
		swap(&buf[start], &buf[j]);
		printf("%li\n", done);
		printArr(buf, frames * 2);
	}
}

