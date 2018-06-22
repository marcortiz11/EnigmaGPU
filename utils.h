#include <iostream>
#include <vector>
#include "math.h"
using namespace std;



inline int numberOfSetBits(int i)
{
     i = i - ((i >> 1) & 0x55555555);
     i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
     return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}

__device__ int dnumberOfSetBits(int i)
{
     i = i - ((i >> 1) & 0x55555555);
     i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
     return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}



void PRINT(vector<int>& sequence){
	for(int i = 0; i<sequence.size()-1; ++i){
		cout << sequence[i] << ",";
	}
	cout << sequence[sequence.size()-1] << endl;
}

void PRINT(vector<vector<int> >& sequence){
 	for(int j = 0;j<sequence.size();++j){
		for(int i = 0; i<sequence[0].size()-1; ++i){
			cout << sequence[j][i] << ",";
		}
		cout << sequence[j][sequence.size()-1] << endl;
	}
}

void PRINT(int* sequence, int size){
	for(int i = 0; i<size-1; ++i){
		cout << sequence[i] << ",";
	}
	cout << sequence[size-1] << endl;
}

__device__ void dPRINT(int* sequence, int size){
	for(int i = 0; i<size-1; ++i){
		printf("%d,",sequence[i]);
	}
	printf("%d\n",sequence[size-1]);
}

__device__ void dPRINT(char* sequence, int size){
	for(int i = 0; i<size-1; ++i){
		printf("%d,",sequence[i]);
	}
	printf("%d\n",sequence[size-1]);
}


__device__ int dmod(int k, int n){
	 return ((k %= n) < 0) ? k+n : k;
}

inline int mod(int k, int n) {
    return ((k %= n) < 0) ? k+n : k;
}

__device__ bool stringcmp(char* a, char* b, int size){
		int i=0;
		while(i<size){
			if(a[i] != b[i]) return false;
			++i;
		}
		return true;
} 

