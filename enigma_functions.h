#include <iostream>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 
#include <algorithm>
#include <utility>
#include <vector>
#include "utils.h"
using namespace std;

int N = 16; //Size of the alphabet to encrypt, N has to be even smaller than 32
 
 
/*typedef struct{
	char *message,
	char *message_encrypted,
	int message_length,
	int *mirror,
	int *disk3, 
	int *disk2, 
	int *disk1,
	char *keyboard;
} crypt_params;
*/

void configureDisk(vector<vector<int> >& disk){ 
	for(int i=0; i<disk.size();++i) disk[i][0] = i;
	random_shuffle(disk.begin(),disk.end());
}


 
void configureDisk(int* disk){
	vector<int> aux(N);
	for(int i=0; i<N;++i) aux[i] = i*2;
	random_shuffle(aux.begin(),aux.end());
	for(int i=0; i<N; i+=1) {
		disk[i*2] = aux[i];
	}
}


void randomPairs(vector<vector<int> >& mirror, int i){ 
	if(i<mirror.size()){
		if(mirror[i][0] == -1){
			bool foundPair = false;
			int j = 0;
			while(!foundPair){
				foundPair = (mirror[j][0] == -1 && rand()%N == 0);
				if(foundPair){
					mirror[j][0] = i;
					mirror[i][0] = j;
				}
				j = (j+1)%mirror.size();
			}
		}
		randomPairs(mirror,i+1);
	}
}
 


void randomPairs(vector<int>& mirror, int i){ 
	if(i<mirror.size()){
		if(mirror[i] == -1){
			bool foundPair = false;
			int j = 0;
			while(!foundPair){
				foundPair = (mirror[j] == -1 && rand()%N == 0);
				if(foundPair){
					mirror[j] = i;
					mirror[i] = j;
				}
				j = (j+1)%mirror.size();
			}
		}
		randomPairs(mirror,i+1);
	}
}

 
void randomPairs(int* mirror, int step, int i){ 
	if(i< (step == 1 ? 6 : N*step)){
		if(mirror[i] == -1){
			bool foundPair = false;
			int j = 0;
			while(!foundPair){
				foundPair = (mirror[j] == -1 && rand()%N == 0);
				if(foundPair){
					mirror[j] = i;
					mirror[i] = j;
				}
				j = (j+step)%(step*N);
			}
		}
		randomPairs(mirror,step,i+step);
	}
}



void join(vector<vector<int> >& diskTop, vector<vector<int> >& diskUnder){
	for (int i = 0; i<diskUnder.size();++i){
		diskTop[diskUnder[i][0]][1] = i;
	}
}



void join(int* diskTop, int* diskUnder){
	for (int i = 0; i<2*N;i+=2){
		diskTop[diskUnder[i]+1] = i;
	}
}



string crypt_cpu(string message,vector<vector<int> >&mirror
						   ,vector<vector<int> >& disk3
						   ,vector<vector<int> >& disk2
						   ,vector<vector<int> >& disk1
						   ,vector<int>& keyboard
						   ,int offset){		   
		string res = "";
		for(int i = 0; i<message.size(); ++i){
			int esymbol = keyboard[message.at(i)-'a'];
			//Forward pass
			esymbol = mod(disk1[mod(esymbol+offset,N)][0] - offset,N);
			esymbol = mod(disk2[mod(esymbol+offset/N,N)][0] - offset/N,N);
			esymbol = mod(disk3[mod(esymbol+offset/N*N,N)][0] - offset/N*N,N);
			esymbol = mirror[esymbol][0];
			//Reflect pass
			esymbol = mod(mirror[mod(esymbol+offset/N*N,N)][1] - offset/N*N,N);
			esymbol = mod(disk3[mod(esymbol+offset/N,N)][1] - offset/N,N);
			esymbol = mod(disk2[mod(esymbol+offset,N)][1] - offset,N);
			esymbol = keyboard[esymbol];
			char encrypted = esymbol + 'a';
			res+=encrypted;
			++offset;
		}	
		return res;
}


string crypt_cpu(string message,int *mirror
						   ,int *disk3
						   ,int *disk2
						   ,int *disk1
						   ,int *keyboard
						   ,int offset){		   
		string res = "";
		int M = N*2;
		offset = offset*2;
		for(int i = 0; i<message.size(); ++i){
			int esymbol = keyboard[message.at(i)-'a']*2;
			//Forward pass
			esymbol = mod(disk1[mod(esymbol+offset,M)] - offset,M);
			esymbol = mod(disk2[mod(esymbol+2*(offset/M),M)] - 2*(offset/M),M);
			esymbol = mod(disk3[mod(esymbol+2*(offset/(2*N*N)),M)] - 2*(offset/(2*N*N)),M);
			esymbol = mirror[esymbol];
			//Reflect pass
			esymbol = mod(mirror[mod(esymbol+2*(offset/(2*N*N)),M) + 1] - 2*(offset/(2*N*N)),M);
			esymbol = mod(disk3[mod(esymbol+2*(offset/M),M) + 1] - 2*(offset/M),M);
			esymbol = mod(disk2[mod(esymbol+offset,M) + 1] - offset,M);
			esymbol = keyboard[esymbol/2];
			char encrypted = esymbol + 'a';
			res+=encrypted;
			offset+=2;
		}	
		return res;
}

//GPU encrypt
