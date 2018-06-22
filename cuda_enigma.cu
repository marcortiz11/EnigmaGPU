#include <iostream>
#include <string>
#include <stdlib.h>
#include <time.h> 
#include <algorithm>
#include <utility>
#include <vector>
#include <ctime>
using namespace std;


int N = 26; //Size of the alphabet to encrypt, N has to be even.
int THREADS = 64;

__device__ int dmod(int k, int n){
	 return ((k %= n) < 0) ? k+n : k;
}

int mod(int k, int n) {
    return ((k %= n) < 0) ? k+n : k;
}

//GPU encrypt
__global__ void crypt_gpu(int msg_length, int N,char *dmessage_encrypted, 
							int *disk3, 
							int *disk2, 
							int *disk1, 
							int *mirror, int offset, int *keyboard){

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i<msg_length){
		int M = N*2;
		offset = (offset + i)*2;
		int esymbol = keyboard[(dmessage_encrypted[i]-'a')*2];
		//Forward pass
		esymbol = dmod(disk1[dmod(esymbol+offset,M)] - offset,M);
		esymbol = dmod(disk2[dmod(esymbol+offset/M,M)] - offset/M,M);
		esymbol = dmod(disk3[dmod(esymbol+offset/M*M,M)] - offset/M,M);
		esymbol = mirror[esymbol];
		//Reflect pass
		esymbol = dmod(mirror[dmod(esymbol+offset/M*M,M) + 1] - offset/M*M,M);
		esymbol = dmod(disk3[dmod(esymbol+offset/M,M) + 1] - offset/M,M);
		esymbol = dmod(disk2[dmod(esymbol+offset,M) + 1] - offset,M);
		esymbol = keyboard[esymbol];
		dmessage_encrypted[i] = esymbol/2 + 'a';
	}
}


string crypt_cpu(string message,int *mirror
						   ,int *disk3
						   ,int *disk2
						   ,int *disk1
						   ,int *keyboard
						   ,int offset){		   
		string res = "";
		int M = N*2;
		for(int i = 0; i<message.size(); ++i){
			int esymbol = keyboard[(message.at(i)-'a')*2];
			//Forward pass
			esymbol = mod(disk1[mod(esymbol+offset,M)] - offset,M);
			esymbol = mod(disk2[mod(esymbol+offset/M,M)] - offset/M,M);
			esymbol = mod(disk3[mod(esymbol+offset/M*M,M)] - offset/M,M);
			esymbol = mirror[esymbol];
			//Reflect pass
			esymbol = mod(mirror[mod(esymbol+offset/M*M,M) + 1] - offset/M*M,M);
			esymbol = mod(disk3[mod(esymbol+offset/M,M) + 1] - offset/M,M);
			esymbol = mod(disk2[mod(esymbol+offset,M) + 1] - offset,M);
			esymbol = keyboard[esymbol];
			char encrypted = esymbol/2 + 'a';
			res+=encrypted;
			offset+=2;
		}	
		return res;
}



void configureDisk(int* disk){
	vector<int> aux(N);
	for(int i=0; i<N;++i) aux[i] = i*2;
	random_shuffle(aux.begin(),aux.end());
	for(int i=0; i<N; i+=1) {
		disk[i*2] = aux[i];
	}
}
void randomPairs(int* mirror, int i){ 
	if(i<N*2){
		if(mirror[i] == -1){
			bool foundPair = false;
			int j = 0;
			while(!foundPair){
				foundPair = (mirror[j] == -1 && rand()%N == 0 && i!=j);
				if(foundPair){
					mirror[j] = i;
					mirror[i] = j;
				}
				j = (j+2)%(2*N);
			}
		}
		randomPairs(mirror,i+2);
	}
}
void join(int* diskTop, int* diskUnder){
	for (int i = 0; i<2*N;i+=2){
		diskTop[diskUnder[i]+1] = i;
	}
}
void CheckCudaError(char sms[], int line);
float GetTime(void);


int main(){

	char message[] = "supercalifragilispicoespialidos";
	srand (time(NULL));
	int numBytes = N*2*sizeof(int);

	cudaEvent_t E0, E1;
	float TiempoTotal;
	float t1,t2;

	cudaEventCreate(&E0);
	cudaEventCreate(&E1);

	//HOST DATA:
	int *hdisk1, *hdisk2, *hdisk3, *hmirror, *hkeyboard;
	char* hmessage_encrypted;
	//host space allocation:
	hdisk1 = (int*) malloc(numBytes);
	hdisk2 = (int*) malloc(numBytes);
	hdisk3 = (int*) malloc(numBytes);
	hmirror = (int*) malloc(numBytes);
	hkeyboard = (int*) malloc(numBytes);
	hmessage_encrypted = (char*) malloc(strlen(message)*sizeof(char));
	//initialization:
	for(int i = 0; i<N*2;++i) hmirror[i] = hdisk3[i] = hdisk2[i] = hdisk1[i] = hkeyboard[i] = -1;
	configureDisk(hdisk3);configureDisk(hdisk2);configureDisk(hdisk1);
	randomPairs(hmirror,0);randomPairs(hkeyboard,0);	
	join(hmirror,hdisk3);
	join(hdisk3,hdisk2);
	join(hdisk2,hdisk1);

	//DEVICE DATA:
	int *ddisk1,*ddisk2,*ddisk3,*dmirror,*dkeyboard;
	char *dmessage_encrypted;

	cudaEventRecord(E0, 0);
	cudaEventSynchronize(E0);

	//device space allocation
	cudaMalloc((int**)&ddisk3, numBytes); 
	cudaMalloc((int**)&ddisk2, numBytes); 
	cudaMalloc((int**)&ddisk1, numBytes); 
	cudaMalloc((int**)&dmirror, numBytes);
	cudaMalloc((int**)&dkeyboard, numBytes);
	cudaMalloc((char**)&dmessage_encrypted,strlen(message)*sizeof(char));
	CheckCudaError((char *) "Obtener Memoria en el device", __LINE__);
	//Copy data host --> device
	cudaMemcpy(ddisk1, hdisk1, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(ddisk2, hdisk2, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(ddisk3, hdisk3, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dmirror, hmirror, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dkeyboard, hkeyboard, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dmessage_encrypted, message, strlen(message)*sizeof(char),cudaMemcpyHostToDevice);
	CheckCudaError((char *) "Copiar Datos Host --> Device", __LINE__); 
	

	int nBlocks = (strlen(message) + THREADS - 1)/THREADS;
	dim3 dimGrid(nBlocks, 1, 1);
	dim3 dimBlock(THREADS, 1, 1);

    int disk_offset = 0;

	// Ejecutar el kernel 
	crypt_gpu<<<dimGrid, dimBlock>>>(strlen(message),N,dmessage_encrypted, ddisk3, ddisk2, ddisk1, dmirror, disk_offset, dkeyboard);
	CheckCudaError((char *) "Invocar Kernel", __LINE__);

	cudaMemcpy(hmessage_encrypted,dmessage_encrypted,strlen(message)*sizeof(char),cudaMemcpyDeviceToHost);
	CheckCudaError((char *) "Copiar Datos Device --> Host", __LINE__);

	cudaEventRecord(E1, 0);
	cudaEventSynchronize(E1);

	cudaEventElapsedTime(&TiempoTotal,  E0, E1);

  	cudaFree(ddisk1); cudaFree(ddisk2); cudaFree(ddisk3); cudaFree(dmirror); cudaFree(dkeyboard); cudaFree(dmessage_encrypted);

  	clock_t begin = clock();
	string res = crypt_cpu(message,hmirror,hdisk3,hdisk2,hdisk1,hkeyboard,disk_offset);
  	clock_t end = clock();
  	t2 = double(end - begin) / CLOCKS_PER_SEC;

    
    string rec = crypt_cpu(res,hmirror,hdisk3,hdisk2,hdisk1,hkeyboard,disk_offset);
  	cout << "Original message: " << message << endl;
  	cout << "Encrypted message CPU: " << res << endl;
  	cout << "Encrypted message GPU: " << hmessage_encrypted << endl;
  	cout << "GPU time to encrypt: " << TiempoTotal << endl;
  	cout << "CPU time to encrypt: " << t2 << endl;

}




void CheckCudaError(char sms[], int line) {
  cudaError_t error;
  error = cudaGetLastError();
  if (error) {
    printf("(ERROR) %s - %s in %s at line %d\n", sms, cudaGetErrorString(error), __FILE__, line);
    exit(EXIT_FAILURE);
  }
}
