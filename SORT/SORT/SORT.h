#ifndef _SORT_
#define _SORT_

template<typename T>
class SORT
{
public:
	
	void Bubble_sort(T &a,int N);
	void Quick_sort(T &a,int l,int r);
	void Initial(T &a,int N);
	void insertSort(T &a,int N);
	void xierSort(T &a,int N);
	void zxSort(T &a,int N);
	bool mergeSort(T &a,int N);
	void makeMinHeap(T &a,int N);
	void minHeapSort(T &a,int N);
	SORT(){};
	~SORT(){};
};




//交换函数
template<typename Type>
void swap(Type &a, Type &b)
{
	// 	Type temp;
	// 	temp=a;
	// 	a=b;
	// 	b=temp;
	if (a!=b)
	{
		a^=b;
		b^=a;
		a^=b;
	}
}




//初始化vector
template<typename T> 
void SORT<T>::Initial(T &a,int N)
{
	for (int i=0;i<N;i++)
	{
		T::value_type temp;
		std::cin>>temp;
		a.push_back(temp);
	}
}


//冒泡排序
template<typename T>
void SORT<T>::Bubble_sort(T &a,int N)
{
	for (int i=0;i<N;i++)
	{
		for (int j=0;j<N-1-i;j++)
		{
			if (a[j]>a[j+1])
			{
				T::value_type t;
				t=a[j+1];
				a[j+1]=a[j];
				a[j]=t;
				//swap(a[j+1],a[j]);
			}
		}
	}
}

//快速排序
template<typename T>
void SORT<T>::Quick_sort(T &a,int l,int r)
{
	if (r<=l)
	{
		return;
	}
	int m=l;
	for (int i=l+1;i<=r;i++)
	{
		if (a[i]<=a[l])
		{

			T::value_type t;
			t=a[++m];
			a[m]=a[i];
			a[i]=t;//swap(&a[++m],&a[i])
		}
	}
	/*swap(&a[l],&a[m]);*/
	T::value_type t;
	t=a[m];
	a[m]=a[l];
	a[l]=t;
	SORT<T>::Quick_sort(a,l,m-1);
	SORT<T>::Quick_sort(a,m+1,r);


}

// 直接插入排序(Insertion  Sort)的基本思想是：每次将一个待排序的记录，按其关
// 键字大小插入到前面已经排好序的子序列中的适当位置，直到全部记录插入完成
// 为止。
template<typename T>
void SORT<T>::insertSort(T &a,int N)
{
	int i,j;
	for (i=1;i<N;i++)
	{
		if (a[i]<a[i-1])
		{
			T::value_type temp;
			temp=a[i];
			for(j=i-1;j>=0&&a[j]>temp;j--)
			{
				a[j+1]=a[j];
			}
			a[j+1]=temp;
		}	

	}
}
//希尔排序
//希尔排序的实质就是分组插入排序，该方法又称缩小增量排序
// 该方法的基本思想是：先将整个待排元素序列分割成若干个子序列（由相隔某个
// “增量”的元素组成的）分别进行直接插入排序，然后依次缩减增量再进行排序，
// 待整个序列中的元素基本有序（增量足够小）时，再对全体元素进行一次直接插
// 入排序。因为直接插入排序在元素基本有序的情况下（接近最好情况），效率是
//很高的，因此希尔排序在时间效率上比前两种方法有较大提高。
template<typename T>
void SORT<T>::xierSort(T &a,int N)
{
	int i,gap,j;
	for (gap=N/2;gap>0;gap/=2)
	{
		for (i=gap;i<N;i++)
		{
			for (j=i-gap;j>=0&&a[j]>a[j+gap];j-=gap)
			{
				T::value_type temp;
				temp=a[j];
				a[j]=a[j+gap];
				a[j+gap]=temp;
			}

		}
	}
}

//直接选择排序是从无序区选一个最小的元素直接放到有序区的最后。
// 设数组为 a[0…n-1]。
//	4.  初始时，数组全为无序区为 a[0..n-1]。令 i=0
//	5.  在无序区 a[i…n-1]中选取一个最小的元素，将其与 a[i]交换。交换之后 a[0…i]
//就形成了一个有序区。
//	6.  i++并重复第二步直到 i==n-1。排序完成。
template<typename T>
void SORT<T>::zxSort(T &a,int N)
{
	int i,j,nmin;
	for (i=0;i<N;i++)
	{
		nmin=i;
		for (j=i+1;j<N;j++)
		{
			if (a[j]<a[nmin])
			{
				nmin=j;
			}
		}
		swap(a[i],a[nmin]);
	}
}

//归并排序
//该算法是采用分治法（Divide and Conquer）的一个非常典型的应用。
//这样通过先递归的分解数列，再合并数列就完成了归并排序
template<typename T,typename Type>
void mergearray(T &a,int first,int mid,int last,Type &tempArray)
{
	int i=first,j=mid+1;
	int k=first;

	while(i<=mid&&j<=last)
	{
		if (a[i]<a[j])
		{
			tempArray[k++]=a[i++];
		}
		else
			tempArray[k++]=a[j++];
	}
	while (i<=mid)
	{
		tempArray[k++]=a[i++];
	}
	while (j<=last)
	{
		tempArray[k++]=a[j++];
	}
	for (int m=0;m<k;m++)
	{
		a[m]=tempArray[m];
	}
}


template<typename T,typename Type>
void mssort(T &a,int first,int last,Type &tempArray)
{
	if (first<last)
	{
		int mid=(first+last)/2;
		mssort(a,first,mid,tempArray);
		mssort(a,mid+1,last,tempArray);
		mergearray(a,first,mid,last,tempArray);
	}
}

template<typename T>
bool SORT<T>::mergeSort(T &a,int N)
{
	T::value_type *tempArray=new T::value_type[N];
	if (tempArray==NULL)
	{
		return false;
	}
	mssort(a,0,N-1,tempArray);
	delete[] tempArray;
	return true;
}

//建立最小堆
template<typename T>
void minHeapDown(T &a,int i,int N)
{
	T::value_type temp;
	temp=a[i];
	int j=2*i+1;
	while (j<N)
	{
		if (j+1<N&&a[j]>a[j+1])
		{
			j++;
		}
		if (a[j]>temp)
		{
			break;
		}
		a[i]=a[j];
		i=j;
		j=2*i+1;
	}
	a[i]=temp;
}

template<typename T>
void SORT<T>::makeMinHeap(T &a,int N)
{
	for (int i=(N-1-1)/2;i>=0;i--)
	{
		minHeapDown(a,i,N);
	}

}

//最小堆排序
template<typename T>
void SORT<T>::minHeapSort(T &a,int N)
{
	for (int i=N-1;i>=0;i--)
	{
		swap(a[0],a[i]);
		minHeapDown(a,0,i);
	}

}

#endif