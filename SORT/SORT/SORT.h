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




//��������
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




//��ʼ��vector
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


//ð������
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

//��������
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

// ֱ�Ӳ�������(Insertion  Sort)�Ļ���˼���ǣ�ÿ�ν�һ��������ļ�¼�������
// ���ִ�С���뵽ǰ���Ѿ��ź�����������е��ʵ�λ�ã�ֱ��ȫ����¼�������
// Ϊֹ��
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
//ϣ������
//ϣ�������ʵ�ʾ��Ƿ���������򣬸÷����ֳ���С��������
// �÷����Ļ���˼���ǣ��Ƚ���������Ԫ�����зָ�����ɸ������У������ĳ��
// ����������Ԫ����ɵģ��ֱ����ֱ�Ӳ�������Ȼ���������������ٽ�������
// �����������е�Ԫ�ػ������������㹻С��ʱ���ٶ�ȫ��Ԫ�ؽ���һ��ֱ�Ӳ�
// ��������Ϊֱ�Ӳ���������Ԫ�ػ������������£��ӽ�����������Ч����
//�ܸߵģ����ϣ��������ʱ��Ч���ϱ�ǰ���ַ����нϴ���ߡ�
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

//ֱ��ѡ�������Ǵ�������ѡһ����С��Ԫ��ֱ�ӷŵ������������
// ������Ϊ a[0��n-1]��
//	4.  ��ʼʱ������ȫΪ������Ϊ a[0..n-1]���� i=0
//	5.  �������� a[i��n-1]��ѡȡһ����С��Ԫ�أ������� a[i]����������֮�� a[0��i]
//���γ���һ����������
//	6.  i++���ظ��ڶ���ֱ�� i==n-1��������ɡ�
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

//�鲢����
//���㷨�ǲ��÷��η���Divide and Conquer����һ���ǳ����͵�Ӧ�á�
//����ͨ���ȵݹ�ķֽ����У��ٺϲ����о�����˹鲢����
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

//������С��
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

//��С������
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