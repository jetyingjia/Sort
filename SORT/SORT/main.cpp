#include <iostream>
//#include <algorithm>
#include <string>
#include <vector>
#include <iterator>
#include "SORT.h"
//#include "SORT.cpp"


void main()
{
	extern int jet;
	
	
	//***********int¿‡–Õ≈≈–Ú******************//
	SORT< std::vector<int> > iSort;
	std::cout<<"Please initial N numbers: ";
	int N;
	std::cin>>N;
	std::vector<int> ivec;

	//≥ı ºªØ
	iSort.Initial(ivec,N);
	std::cout<<"The numbers are: ";
	for (int i=0;i<N;i++)
	{
		std::cout<<ivec[i]<<" ";
	}

// 	//√∞≈›≈≈–Ú
// 	std::vector<int> Bivec(ivec.begin(),ivec.end());
// 	iSort.Bubble_sort(Bivec,N);
// 	std::cout<<std::endl<<"The Bubble_sort is: ";
// 	for (int i=0;i<N;i++)
// 	{
// 		std::cout<<Bivec[i]<<" ";
// 	}
// 
// 	//øÏÀŸ≈≈–Ú
// 	std::vector<int> Qivec(ivec.begin(),ivec.end());
// 	iSort.Quick_sort(Qivec,0,N-1);
// 	std::cout<<std::endl<<"The Quick_sort is: ";
// 	for (int i=0;i<N;i++)
// 	{
// 		std::cout<<Qivec[i]<<" ";
// 	}
// 
// 	//÷±Ω”≤Â»Î≈≈–Ú
// 	std::vector<int> Zivec(ivec.begin(),ivec.end());
// 	iSort.insertSort(Zivec,N);
// 	std::cout<<std::endl<<"The insertSort is: ";
// 	for (int i=0;i<N;i++)
// 	{
// 		std::cout<<Zivec[i]<<" ";
// 	}
// 	//œ£∂˚≈≈–Ú
// 	std::vector<int> xiivec(ivec.begin(),ivec.end());
// 	iSort.xierSort(xiivec,N);
// 	std::cout<<std::endl<<"xier sort: ";
// 	for (int i=0;i<N;i++)
// 	{
// 		std::cout<<xiivec[i];
// 	}
// 	//÷±Ω”—°‘Ò≈≈–Ú
// 	std::vector<int> zxivec(ivec.begin(),ivec.end());
// 	iSort.zxSort(zxivec,N);
// 	std::cout<<std::endl<<"zhijie xuanze sort: ";
// 	for (int i=0;i<N;i++)
// 	{
// 		std::cout<<zxivec[i]<<" ";
// 	}
	//πÈ≤¢≈≈–Ú
// 	std::vector<int> mivec(ivec.begin(),ivec.end());
// 	iSort.mergeSort(mivec,N);
// 	std::cout<<std::endl<<"mergeSort: ";
// 	for (int i=0;i<N;i++)
// 	{
// 		std::cout<<mivec[i]<<" ";
// 	}

	//Ω®¡¢◊Ó–°∂—
	std::vector <int> Hivec(ivec.begin(),ivec.end()); 
	iSort.makeMinHeap(Hivec,N);
	std::cout<<std::endl<<"Make Min Heap: ";
	for (int i=0;i<N;i++)
	{
		std::cout<<Hivec[i]<<" ";
	}
// 	//∂—≈≈–Ú
	std::cout<<std::endl<<"Min Heap Sort: ";
	iSort.minHeapSort(Hivec,N);
	for (int i=0;i<N;i++)
	{
		std::cout<<Hivec[i]<<" ";
	}

	
	

	//***********string¿‡–Õ≈≈–Ú*************//
	SORT< std::vector<std::string> > sSort;
	std::cout<<"Please initial M strings: ";
	int M;
	std::cin>>M;
	std::vector<std::string> svec;

	//≥ı ºªØ
	sSort.Initial(svec,M);
	std::cout<<"The strings are: ";
	for (int i=0;i<M;i++)
	{
		std::cout<<svec[i]<<" ";
	}

// 	//√∞≈›≈≈–Ú
// 	std::vector<std::string> sBivec(svec.begin(),svec.end());
// 	sSort.Bubble_sort(sBivec,M);
// 	std::cout<<std::endl<<"The Bubble_sort is: ";
// 	for (int i=0;i<M;i++)
// 	{
// 		std::cout<<sBivec[i]<<" ";
// 	}
// 
// 	//øÏÀŸ≈≈–Ú
// 	std::vector<std::string> sQivec(svec.begin(),svec.end());
// 	sSort.Quick_sort(sQivec,0,M-1);
// 	std::cout<<std::endl<<"The Quick_sort is: ";
// 	for (int i=0;i<M;i++)
// 	{
// 		std::cout<<sQivec[i]<<" ";
// 	}
// 
// 	//÷±Ω”≤Â»Î≈≈–Ú
// 	std::vector<std::string> Zsvec(svec.begin(),svec.end());
// 	sSort.insertSort(Zsvec,M);
// 	std::cout<<std::endl<<"The insertSort is: ";
// 	for (int i=0;i<M;i++)
// 	{
// 		std::cout<<Zsvec[i]<<" ";
// 	}
// 	//œ£∂˚≈≈–Ú
// 	std::vector<std::string> xisvec(svec.begin(),svec.end());
// 	sSort.xierSort(xisvec,M);
// 	std::cout<<std::endl<<"xier sort: ";
// 	for (int i=0;i<M;i++)
// 	{
// 		std::cout<<xisvec[i]<<" ";
// 	}
// 	//÷±Ω”—°‘Ò≈≈–Ú
// 	std::vector<std::string> zxsvec(svec.begin(),svec.end());
// 	sSort.zxSort(zxsvec,M);
// 	std::cout<<std::endl<<"zhijie xuanze sort: ";
// 	for (int i=0;i<M;i++)
// 	{
// 		std::cout<<zxsvec[i]<<" ";
// 	}
	//πÈ≤¢≈≈–Ú
// 	std::vector<std::string> msvec(svec.begin(),svec.end());
// 	sSort.mergeSort(msvec,N);
// 	std::cout<<std::endl<<"mergeSort: ";
// 	for (int i=0;i<N;i++)
// 	{
// 		std::cout<<msvec[i]<<" ";
// 	}

	system("pause");

}