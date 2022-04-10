#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h> 
#include <thrust/gather.h>
#include <thrust/functional.h>
#include <thrust/tuple.h>
#include <thrust/reduce.h>
#include <thrust/random.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/binary_search.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/transform_reduce.h>
#include <thrust/remove.h>
#include <thrust/set_operations.h>
#include <limits>
#include <iterator>
#include <fstream>
#include <iomanip> 
#include <iostream>   
#include <thrust/extrema.h>
#include <thrust/unique.h>
#include <string>
#include <thrust/count.h>  
#include <thrust/merge.h>

using namespace thrust::placeholders;   
__global__ void mykernel(float* d_out,float* d_in,int rows,int cols,int k){
    int idx=threadIdx.x+blockIdx.x*blockDim.x;
    while (idx < rows*cols) {
      d_out[idx]=d_in[idx]/d_in[idx%rows+k*rows];
      idx+= gridDim.x*blockDim.x;
    }
}
struct s_rat
{
  template <typename T1, typename T2>
  __host__ __device__
  bool operator()(const T1 &t1, const T2 &t2){
    if  (thrust::get<1>(t1) < thrust::get<1>(t2)) return true;
    if  (thrust::get<1>(t1) > thrust::get<1>(t2)) return false;
    if  (thrust::get<0>(t1) < thrust::get<0>(t2)) return true;
    return false;
  }
};
 
struct lb 
{  
    const float tot;
    lb(float _tot): tot(_tot){}
    __host__ __device__ 
   float operator()(float& x) {
   return tot-2.0*x;
  }
};
 
 struct lb1 
{  
    const float tot;
    lb1(float _tot): tot(_tot){}
    __host__ __device__ 
   float operator()(float& x) {
   return 2.0*x-tot;
  }
};

 struct lbs
{  
    __host__ __device__ 
   float operator()(float& x,float& y) {
   return x+2*y;
  }
};

struct is_neg
{
  __host__ __device__
  bool operator()(const float x)
  {
    return (signbit(x) || x==0.0);
  }
};

struct is_pos
{
  __host__ __device__
  bool operator()(const float x)
  {
    return (signbit(-x));
  }
};

template <typename T>
struct absv
{
  __host__ __device__ T operator()(const T &x) const
  {
    return (x < T(0)) ? -x : x;
  }

};

struct is_k
{
  const int tot;
    is_k(int _tot): tot(_tot){}
  __host__ __device__
  bool operator()(const int x)
  {
    return (x == tot);
  }
};


template <typename Iterator>
void print_range(Iterator first, Iterator last)
{
    typedef typename std::iterator_traits<Iterator>::value_type T;
 
    thrust::copy(first, last, std::ostream_iterator<T>(std::cout<< std::setw(6) << std::fixed<< std::setprecision(1), " "));  
    std::cout << "\n";
}


template <typename Iterator>
void writefile(Iterator first, Iterator last)
{
    typedef typename std::iterator_traits<Iterator>::value_type T;
    std::ofstream output;
	  output.open("breakpoints"); 
    thrust::copy(first, last, std::ostream_iterator<T>(std::cout<< std::setw(6) << std::fixed<< std::setprecision(1), " "));  
    
}

void rc_find(FILE *fp,int* rows,int* cols)
{    
    *rows = 0;
    int i,j;
    *cols = 0; 
     
    while((i=fgetc(fp))!=EOF)
    {
            if (i == ' ') {
                ++j;
            } 
            else if (i == '\n') {
                (*rows)++; 
                *cols=j+1;
                j = 0;   
            }
    } 
    fclose(fp);
}

int main(int argc,char **argv){   
    
    char *problem  = (char *) malloc ((100) * sizeof (char));
    strcpy(problem,argv[1]);
    float *h_in, *d_in,*d_out;   
      
    FILE *getrc = fopen(problem,"r");  
    int rows, cols; 
    rc_find(getrc,&rows,&cols);
    int N = rows*cols;
 
      
    h_in = (float *) malloc (N*sizeof (float));
    FILE *data = fopen(problem,"r");    
    for (int j=0;j<rows;j++) {
      for (int i=0;i<cols;i++){ 
        fscanf(data, "%f",&h_in[i*rows+j]);
    }}
    fclose(data);
    
    cudaMalloc((void **) &d_in,N*sizeof(float));
    cudaMalloc((void **) &d_out,N*sizeof(float));
    
    cudaMemcpy(d_in,h_in,N*sizeof(float),cudaMemcpyHostToDevice);
    
     
    typedef thrust::device_vector<float> Vector;
    
    float minupp=99999999;
    float maxlow=0;
    float finalavg=0;
    float finalsize=0;
    for (int k = 0;k<cols;k++) {
    mykernel<<<128,128>>>(d_out,d_in,rows,cols,k); 
  
    
    thrust::device_vector<float> ratio(d_out, d_out+N);
    thrust::device_vector<float> xjhat(d_in,d_in+N);     
    thrust::device_vector<float> l_lamb(N);
    thrust::device_vector<float> r_lamb(N); 
    thrust::device_vector<int> index(ratio.size());
    thrust::device_vector<int> jhat(ratio.size());
    thrust::sequence(index.begin(), index.end());
    thrust::sequence(jhat.begin(), jhat.end());
    thrust::transform(index.begin(), index.end(), index.begin(), _1/rows);
    thrust::transform(jhat.begin(), jhat.end(), jhat.begin(), _1%rows+k*rows);
    auto myit = thrust::make_zip_iterator(thrust::make_tuple(ratio.begin(), index.begin(), jhat.begin()));
    thrust::sort(myit, myit+N, s_rat()); 
    thrust::gather(thrust::device,jhat.begin(), jhat.end(),xjhat.begin(),xjhat.begin());
    thrust::device_vector<float> abs_xjhat(N);
    abs_xjhat = xjhat;
    thrust::device_vector<float> inc(N); 
    thrust::device_vector<float> exc(N); 
    thrust::transform(abs_xjhat.begin(),abs_xjhat.end(),abs_xjhat.begin(),absv<float>()); 
    thrust::exclusive_scan_by_key(index.begin(), index.end(), abs_xjhat.begin(), exc.begin(),0.0,thrust::equal_to<int>(),thrust::plus<float>());
    thrust::inclusive_scan_by_key(index.begin(), index.end(), abs_xjhat.begin(), inc.begin(),thrust::equal_to<int>(),thrust::plus<float> ());
    thrust::transform(inc.begin(),inc.end(),l_lamb.begin(),lb(thrust::reduce(abs_xjhat.begin(),abs_xjhat.begin()+rows)));
    thrust::transform(exc.begin(),exc.end(),r_lamb.begin(),lb1(thrust::reduce(abs_xjhat.begin(),abs_xjhat.begin()+rows)));
     
    thrust::device_vector<int> se(ratio.size()); 
    thrust::device_vector<float> lambdas(ratio.size()); 
    thrust::sequence(se.begin(), se.end());
    thrust::gather_if(se.begin(),se.end(),ratio.begin(),r_lamb.begin(),l_lamb.begin(),is_neg()); 
     
    
    typedef Vector::iterator           Iterator;
    thrust::transform(l_lamb.begin(),l_lamb.end(),abs_xjhat.begin(),lambdas.begin(),lbs()); 
    Vector r1(ratio.size());
    thrust::remove_copy_if(lambdas.begin(), lambdas.end(), index.begin(),r1.begin(), is_k(k)); 
    Iterator iter = thrust::remove_if(r1.begin(), r1.end(),is_neg());
    r1.resize(iter-r1.begin());
    Iterator iter2 = thrust::unique(r1.begin(),r1.end());
    r1.resize(iter2-r1.begin());  

    typedef thrust::device_vector<float>::iterator ft;
    ft min = min_element(r1.begin(), r1.end()); 
    ft max = max_element(r1.begin(), r1.end());
    if (*min < minupp) minupp = *min;
    if (*max > maxlow) maxlow = *max; 
    float avg = reduce(r1.begin(),r1.end());
    float sam = r1.size();
    finalavg= avg + finalavg;
    finalsize=sam +finalsize;
    }
    std::cout << minupp << " " << maxlow << " " << finalavg/finalsize << std::endl;
     
     
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in); 
    return 0;
     
}
