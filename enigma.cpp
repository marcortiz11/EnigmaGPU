#include <iostream>
#include "enigma_functions.h"
using namespace std;

/*
 * 
 * EXEPMLE D'UNA MÀQUINA ENIGMA EN C++
 * 
 */

int main(){
	//The enigma machine had 3 disks and a mirroring mechanism

	vector<vector<int> >disk1(N,vector<int>(2,-1)),
						disk2(N,vector<int>(2,-1)),
						disk3(N,vector<int>(2,-1)),
						mirror(N,vector<int>(2,-1));

	vector<int> keyboard(N,-1);
	
	srand (time(NULL));
	
	//Offset of disk1
	int disk_offset = 0;
	
	//Creating the connections:
	configureDisk(disk1);
	configureDisk(disk2);
	configureDisk(disk3);
	randomPairs(mirror,0);
	randomPairs(keyboard,0);
	
	join(mirror,disk3);
	join(disk3,disk2);
	join(disk2,disk1);

	//Text a encriptar:
	/*string message = "supercalifragilisticoespialidoso";
	string encrypted = crypt_cpu(message,mirror,disk3,disk2,disk1,keyboard,disk_offset);
	disk_offset = 0; //Posem la màquina al mateix offset
	string decrypted = crypt_cpu(encrypted,mirror,disk3,disk2,disk1,keyboard,disk_offset);
	cout << "Original: n" << message << endl;
	cout << "Encrypted: " << encrypted << endl;
	cout << "Decripted: " << decrypted << endl;
	*/
	string message;
	while(cin >> message){
		string res = crypt_cpu(message,mirror,disk3,disk2,disk1, keyboard, disk_offset);
		string original = crypt_cpu(res,mirror,disk3,disk2,disk1, keyboard, disk_offset);
		++disk_offset;
		cout << "Original message: " << message << endl;
		cout << "Encrypted message: " << res << endl;
		cout << "Reconstructed message: " << original << endl;
	}
	
}




