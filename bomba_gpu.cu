#include <ctime>
#include <cuda_profiler_api.h>
#include "enigma_functions.h"
using namespace std;
 
// The X dimension of the grid is for Keyboard combinations
int THREADSX = 32;
int BLOCKSX = 256;

//The Y dimension of the grid is for disk start positions
int THREADSY = 8; 
int BLOCKSY = 128;
 

__device__ bool found = false;
__device__ const int dN = 16;
__device__ const int aaa = 32*8;

__device__ char* crypt_gpu( int message_length,
							int *mirror,
							int *disk3, 
							int *disk2, 
							int *disk1, char* message, char *keyboard, int offset){
	int M = dN*2;
	offset = offset*2;
	char *res = (char*) malloc(message_length*sizeof(char));
	for(int i = 0; i<message_length; ++i){
		int esymbol = keyboard[message[i]-'a']*2;
		//Forward pass
		esymbol = dmod(disk1[dmod(esymbol+offset,M)] - offset,M);
		esymbol = dmod(disk2[dmod(esymbol+2*(offset/M),M)] - 2*(offset/M),M);
		esymbol = dmod(disk3[dmod(esymbol+2*(offset/(2*dN*dN)),M)] - 2*(offset/(2*dN*dN)),M);
		esymbol = mirror[esymbol];
		//Reflect pass
		esymbol = dmod(mirror[dmod(esymbol+2*(offset/(2*dN*dN)),M) + 1] - 2*(offset/(2*dN*dN)),M);
		esymbol = dmod(disk3[dmod(esymbol+2*(offset/M),M) + 1] - 2*(offset/M),M);
		esymbol = dmod(disk2[dmod(esymbol+offset,M) + 1] - offset,M);
		esymbol = keyboard[esymbol/2];
		res[i] = esymbol + 'a';
		offset+=2;
	}
	return res;
} 


//Prevents wrong keyboard layouts to be checked in the future
__device__ bool iscompatibleKeyboard(char* message, char* message_encrypted, int message_length,
							int *mirror,
							int *disk3, 
							int *disk2, 
							int *disk1, 
							char* inc_keyboard, int offset){
	char* decrypted = crypt_gpu(message_length, mirror,disk3,disk2,disk1,message_encrypted,inc_keyboard,offset);
	int c=0;
	while(c < message_length){
		int symbol = message_encrypted[c] - 'a';
		if(	inc_keyboard[symbol] != -1 && 
			decrypted[c] != '`' && decrypted[c] != message[c]){	
			free(decrypted);
			return false;
		}
		++c;
	}
	free(decrypted);
	return true; 
}


__device__ void dINCREMENT(char* pila, int numSetBits){
		int carry = 0;
		int left = 1;
		pila[0]++;
		for(int i = 0; i<numSetBits/2; ++i,left+=2){
			 pila[i] = (pila[i]+carry);
			 if(pila[i] == left){
				pila[i] = 0;
				carry = 1;
			 }
		}
}


__device__ void completeKeyboardIter(char* message, char* message_encrypted, int message_length,
								int *mirror,
								int *disk3, 
								int *disk2, 
								int *disk1, 
								char *inc_keyboard, char* pila, int numSetBits, int offset){
	
	if(iscompatibleKeyboard(message,message_encrypted, message_length,mirror,disk3,disk2,disk1,inc_keyboard,offset)){					
		//Conuter to 0							
		for(int i=0; i<dN/2; ++i) pila[i] = 0;
		
		//How many combinations
		int comb = 1;
		for(int i=1; i<numSetBits; i+=2) comb *= (numSetBits-i); // 1 
		bool isCompatible;
		
		//For each swap combination:
		for(int c=0; c<comb && !found;++c){
			int realj=0;
			for(int j=(numSetBits/2)-1; j>=0; --j){
				while(realj<dN && inc_keyboard[realj]!=-1) ++realj;
				int reali=realj+1;
				for(int i=0; i<=pila[j]; ++i){
					while(reali<dN && inc_keyboard[reali]!=-1) ++reali;
				 	++reali;
				} 
				--reali;
				inc_keyboard[realj] = reali; inc_keyboard[reali] = realj;
			}
			isCompatible = iscompatibleKeyboard(message,message_encrypted, message_length,mirror,disk3,disk2,disk1,inc_keyboard,offset);
			if(isCompatible){
				printf("--------------\n");
				printf("SOLUTION:\n");
				printf("keyboard: ");
				dPRINT(inc_keyboard,dN);
				printf("offset: %d\n",offset);
				found = true;
				return;
			}	
			for(int i = 0; i<dN; ++i) inc_keyboard[i] = (inc_keyboard[i] != i) ? -1 : i;
			dINCREMENT(pila,numSetBits);
		}
	}
	
}

  

__global__ void breakEnigma(char* message, char* message_encrypted, int message_length,
							int *mirror,
							int *disk3,
					 		int *disk2,
							int *disk1,
							int *solution_keyboard){

	//Keyboard a shared memory
	__shared__ char inc_keyboard[aaa][dN];
	__shared__ char pila[aaa][dN/2];

	//Number of keyboard layots to combine for the thread:
	int keyboard_comb_perThread = (1<<dN)/(gridDim.x*blockDim.x);
	int keyboard_id = blockIdx.x*blockDim.x+threadIdx.x;
		keyboard_id*=(keyboard_comb_perThread);
 
	//Number of disk starts to combine for the thread:
	int diskStart_perThread = (dN*dN*dN)/(gridDim.y*blockDim.y);
	int start = threadIdx.y+blockIdx.y*blockDim.y;
		start *= diskStart_perThread;

	int tid = threadIdx.x+threadIdx.y*blockDim.x;
	for(int s=start; s<start+diskStart_perThread && !found; ++s){
		for(int c=keyboard_id; c<=keyboard_id+keyboard_comb_perThread && !found; ++c){
			int nBits = dnumberOfSetBits(c);
			if(!(nBits&1) && nBits <= 12){
				for(int i=0;i<dN; ++i){
					inc_keyboard[tid][i] = (c&(1<<i)) ? -1 : i ;
				}
				completeKeyboardIter(message,message_encrypted,message_length,mirror,disk3,disk2,disk1,(char*)&inc_keyboard[tid][0],(char*)&pila[tid][0],nBits,s);
			}
		}
	}

}

 

void CheckCudaError(char sms[], int line);
float GetTime(void);


int main(){ 

	srand (time(NULL));
	int numBytes = N*2*sizeof(int);

	cudaEvent_t E0, E1;
	float TiempoTotal;
 
	cudaEventCreate(&E0);
	cudaEventCreate(&E1);
 
	//HOST DATA: 
	int *hdisk1, *hdisk2, *hdisk3, *hmirror, *hkeyboard;
	int disk_offset = 100;
	string message = "abcdefghijklmnop";
	string message_encrypted;
	int messageLength = message.size();
 
	//host spa ce allocation:
	hdisk1 = (int*) malloc(numBytes);
	hdisk2 = (int*) malloc(numBytes);
	hdisk3 = (int*) malloc(numBytes);
	hmirror = (int*) malloc(numBytes);
	hkeyboard = (int*) malloc(numBytes/2);
  
	//initialization of the data structures:
	for(int i=0; i<N*2;++i) hmirror[i] = hdisk3[i] = hdisk2[i] = hdisk1[i] = hkeyboard[i/2] = -1;
	configureDisk(hdisk3);configureDisk(hdisk2);configureDisk(hdisk1);
	randomPairs(hmirror,2,0);
	randomPairs(hkeyboard,1,0);
	
	for(int i = 0; i<N; ++i) if(hkeyboard[i] == -1) hkeyboard[i] = i;
	  
	join(hmirror,hdisk3);
	join(hdisk3,hdisk2);
	join(hdisk2,hdisk1);
    
	
	message_encrypted = crypt_cpu(message,hmirror,hdisk3,hdisk2,hdisk1,hkeyboard,disk_offset);
	cout << "Message model: " << message << endl;
	cout << "Encrypted message: " << message_encrypted << endl;
	cout << "Reconstructed message : " << crypt_cpu(message_encrypted,hmirror,hdisk3,hdisk2,hdisk1,hkeyboard,disk_offset) << endl;
	cout << "Original Keyboard : ";
	PRINT(hkeyboard,N);
  
	//DEVICE DATA:
	int *ddisk1,*ddisk2,*ddisk3,*dmirror,*dkeyboard;
	char *dmessage,*dmessage_encrypted;
 
	//device space allocation
	cudaMalloc((int**)&ddisk3, numBytes); 
	cudaMalloc((int**)&ddisk2, numBytes); 
	cudaMalloc((int**)&ddisk1, numBytes); 
	cudaMalloc((int**)&dmirror, numBytes);
	cudaMalloc((int**)&dkeyboard, numBytes/2);
	cudaMalloc((char**)&dmessage,message.size()*sizeof(char));
	cudaMalloc((char**)&dmessage_encrypted,message.size()*sizeof(char));
	CheckCudaError((char *) "Obtener Memoria en el device", __LINE__);
 
	//Copy data host --> device
	cudaMemcpy(ddisk1, hdisk1, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(ddisk2, hdisk2, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(ddisk3, hdisk3, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dmirror, hmirror, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dmessage, &message[0u], message.size()*sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(dmessage_encrypted, &message_encrypted[0u], message.size()*sizeof(char), cudaMemcpyHostToDevice);
	CheckCudaError((char *) "Copiar Datos Host --> Device", __LINE__); 

	dim3 dimGrid(BLOCKSX, BLOCKSY, 1);
	dim3 dimBlock(THREADSX, THREADSY, 1);
	
	
	cudaEventRecord(E0, 0);
	cudaEventSynchronize(E0);

	//Ejecutar el kernel 
	breakEnigma<<<dimGrid, dimBlock>>>(dmessage,dmessage_encrypted, N, dmirror, ddisk3, ddisk2, ddisk1, dkeyboard);
	CheckCudaError((char *) "Invocar Kernel", __LINE__);
	
	cudaEventRecord(E1, 0);
	cudaEventSynchronize(E1);
	cudaEventElapsedTime(&TiempoTotal,  E0, E1);
	
	
	cout << "Temps Kernel: " << TiempoTotal << endl;


	//Copiar: 
	//Keyboard
	//cudaMemcpy(hkeyboard,dkeyboard,numBytes/2,cudaMemcpyDeviceToHost);
	//Offset
	//Disk Order
	CheckCudaError((char *) "Copiar Datos Device --> Host", __LINE__);

	//Free GPU memory
  	cudaFree(ddisk1); cudaFree(ddisk2); cudaFree(ddisk3); cudaFree(dmirror),cudaFree(dkeyboard);
  	//Do whatever

  	//Free CPU memory
  	free(hdisk1);free(hdisk2);free(hdisk3); free(hmirror);free(hkeyboard);

}




void CheckCudaError(char sms[], int line) {
  cudaError_t error;
  error = cudaGetLastError();
  if (error) {
    printf("(ERROR) %s - %s in %s at line %d\n", sms, cudaGetErrorString(error), __FILE__, line);
    exit(EXIT_FAILURE);
  }
}
