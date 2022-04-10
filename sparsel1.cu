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
#include <thrust/inner_product.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/transform_reduce.h>
#include <thrust/set_operations.h>
#include <limits>
#include <iterator>
#include <fstream>
#include <iomanip> 
#include <iostream>   
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <istream>

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
  bool operator()(const T1 &t1, const T2 &t2)
  {
    if  (thrust::get<1>(t1) < thrust::get<1>(t2)) return true;
    if  (thrust::get<1>(t1) > thrust::get<1>(t2)) return false;
    if  (thrust::get<0>(t1) < thrust::get<0>(t2)) return true;
    return 0;
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
 
typedef thrust::tuple<int,float,float,float> floatTup;
struct int_pred
{
  const float reg;
  const float tot;
  int_pred(float reg,float tot): reg(reg),tot(tot){} 
  __host__ __device__
   bool operator() (const floatTup& tup )
    {
        const float x = thrust::get<1>(tup); 
        const float y = thrust::get<2>(tup);
        const float z = thrust::get<3>(tup);
        return (z<0? (!(reg <= -x && -y< reg)) : (!(y >=reg && x < reg)));  
    }
};


struct xminusvx
    {
        const float *m_vec1;
        const float *m_vec2;
        const float *m_A; 
        float *m_result;
        size_t v1size;
        xminusvx(thrust::device_vector<float> const& A,thrust::device_vector<float> const& vec1, thrust::device_vector<float> const& vec2,thrust::device_vector<float>& result)
        {
          m_vec1 = thrust::raw_pointer_cast(vec1.data());
          m_vec2 = thrust::raw_pointer_cast(vec2.data()); 
          m_result = thrust::raw_pointer_cast(result.data());
          m_A=thrust::raw_pointer_cast(A.data());
          v1size = vec1.size();
        }

        __host__ __device__
        void operator()(const size_t  x) const
        {
            size_t i = x%v1size;
            size_t j = x/v1size;  
            m_result[i + j * v1size] = fabs(m_A[i+j*v1size] - m_vec1[i] * m_vec2[j]);
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
    float elapsed=0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    char *problem  = (char *) malloc ((100) * sizeof (char));
    strcpy(problem,argv[1]);
    int lamb = atof(argv[2]); 
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
    float zopt = 9999999999999;
    int alpha;
    thrust::device_vector<float> vopt(cols);

    for (int k = 0;k<cols;k++) { 
    mykernel<<<128,128>>>(d_out,d_in,rows,cols,k); 
   
    thrust::device_vector<float> ratio(d_out, d_out+N);  
    thrust::device_vector<float> xjhat(d_in,d_in+N);     
    thrust::device_vector<float> l_lamada(N);
    thrust::device_vector<float> r_lamada(N); 
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
    thrust::transform(inc.begin(),inc.end(),l_lamada.begin(),lb(thrust::reduce(abs_xjhat.begin(),abs_xjhat.begin()+rows)));
    thrust::transform(exc.begin(),exc.end(),r_lamada.begin(),lb(thrust::reduce(abs_xjhat.begin(),abs_xjhat.begin()+rows)));
 
    typedef thrust::device_vector<float>::iterator fit;
    typedef thrust::device_vector<int>::iterator iit;
    typedef thrust::tuple<iit,fit,fit,fit> tup;
    typedef thrust::zip_iterator<tup>  zip_it;
    zip_it v = thrust::remove_if(thrust::make_zip_iterator(thrust::make_tuple(index.begin(),l_lamada.begin(),r_lamada.begin(),ratio.begin())),
                                        thrust::make_zip_iterator(thrust::make_tuple(index.end(),l_lamada.end(),r_lamada.end(),ratio.end())),
                                        int_pred(lamb,thrust::reduce(abs_xjhat.begin(),abs_xjhat.begin()+rows)));
              
                               
    tup endTuple = v.get_iterator_tuple();  
    index.erase(thrust::get<0>(endTuple),index.end());
    l_lamada.erase(thrust::get<1>(endTuple),l_lamada.end());
    r_lamada.erase(thrust::get<2>(endTuple),r_lamada.end());
    ratio.erase( thrust::get<3>(endTuple),ratio.end());

    thrust::device_vector<float>vstar(cols);   
    thrust::device_vector<int>v_keys(cols);  
    thrust::sequence(v_keys.begin(),v_keys.end()); 
    thrust::set_union_by_key(index.begin(),index.end(),v_keys.begin(),v_keys.end(),ratio.begin(),vstar.begin(),v_keys.begin(),vstar.begin()); 
       
    thrust::device_vector<float> vx(N);  
    thrust::device_vector<float> vec1(d_in+k*rows,d_in+k*rows+rows);  
    thrust::device_vector<float> A(d_in,d_in+N);
    thrust::for_each_n(thrust::device, thrust::counting_iterator<size_t>(0),(N),xminusvx(A,vec1,vstar,vx));
    float z = thrust::reduce(vx.begin(),vx.end()) + lamb*thrust::transform_reduce(vstar.begin(),vstar.end(),absv<float>(),0.0,thrust::plus<float>());
   
    if (z <= zopt) {
        zopt = z;
        vopt = vstar;
        alpha = k;	
    }
    }
    float norm = std::sqrt(thrust::inner_product(vopt.begin(),vopt.end(),vopt.begin(),0.0f));
    thrust::transform(vopt.begin(),vopt.end(),vopt.begin(),_1/=norm);

    std::string s = "v_";  
    s.append(&problem[strlen(problem)-1]);
    s.append(argv[2]);
    std::ofstream output;
	  output.open(s);
	  output << alpha+1 << " " << std::endl;
	  for(int i=0;i<cols;++i){
	  output<<vopt[i] << " "  <<std::endl;
  	}
	  output.close();
	  

 

    cudaEventRecord(stop, 0);
    cudaEventSynchronize (stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    std::cout  << elapsed; std::cout << '\n'; 

    free(h_in);
    cudaFree(d_in);
    cudaFree(d_out); 
    
    return 0;
  }
 
