**题目**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201109195405109.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg0OTc2Mw==,size_16,color_FFFFFF,t_70#pic_center)


**题解一：大顶堆**

在n个数[$x_1,x_2,...,x_n$]中，最小的k个数，可能会有以下两种情况

- 就是前n-1个数中最小的k个数  [$a_1,a_2,...,a_k$]

- 若第n个数 $x_n$ 小于前n-1个数中最小的k个数[$a_1,a_2,...,a_k$]中的$a_m$，那么前n个数中最小的k个数可以表示为

  [$a_1,a_2,...x_n,...,a_k$]

按照这种思想，我们就可以先将数组的前k个值看做前k个最小值，并放入优先队列中，通过比较后一个值与队列中最大值的大小，得到前n个数的k个最小值。

```c++
class Solution {
public:
    vector<int> getLeastNumbers(vector<int>& arr, int k) {
        vector<int> minK(k,0);
        if(k == 0) return minK;
        else{
            priority_queue<int> Q;
            //将前k个数放进优先队列中
            for(int i = 0; i < k; i++){
                Q.push(arr[i]);
            }
            //去掉优先队列中最大的值
            for(int i = k; i < (int)arr.size(); i++){
                if(arr[i] < Q.top()){
                    Q.pop();
                    Q.push(arr[i]);
                }
            }
            //将队列中最终的值赋给vector
            for(int i = 0; i < k; i++){
                minK[i] = Q.top();
                Q.pop();
            }
        }
        return minK;
    }
};
```

**时间复杂度** ：$O(n\log{k})$，优先队列的复杂度为$O(\log {k})$，遍历数组的复杂度为$O(n)$。

**空间复杂度**：$O(k)$，优先队列（大根堆）的容量为k。

---
---
---
**题解二：快速排序思想**

1. 对数组进行一轮快速排序，split(arr,low,high,k)得到数组分割点$m$

2. 对于m左边的数组，有如下操作：

   - m - low + 1 == k, 那么已经得到了答案
   - m - low + 1 > k，那么左边的数肯定包含TopK的最小值，在左边继续寻找，quick_sort(arr, low, m - 1, k);
   - m - low + 1 < k，那么左边的数肯定都是TopK的最小值，在右边继续寻找，quick_sort(arr, m + 1, high, k - m + low - 1);

   ```c++
   class Solution {
   public:
       int split(vector<int> &arr, int low, int high){
           int x = arr[low]; //比较基准就是数组的第一个值
           int split_position = low; //比较基准所在位置
   
           for(int i = low + 1; i <= high; i++){
               if (arr[i] <= x)
               {
                   split_position ++;
                   swap(arr[split_position], arr[i]);
               }
           }
           if(low != split_position){
               swap(arr[low], arr[split_position]);    //把基准放到合适的位置
           }
           return split_position;
       }
   
       void quick_sort(vector<int> &arr, int low, int high, int k){
           if(low < high){
               int split_position = split(arr, low, high);
               if (split_position - low + 1 < k){
                   quick_sort(arr, split_position + 1, high, k - split_position + low -1);
               }
               else if (split_position - low + 1 > k)
               {
                   quick_sort(arr, low, split_position - 1, k);
               }
               else if (split_position - low + 1 == k){
                   return ;
               }
           }
       }
   
       vector<int> getLeastNumbers(vector<int>& arr, int k) {
           vector<int> minK(k,0);
           quick_sort(arr, 0, arr.size() - 1, k);
           for(int i = 0; i < k; i++){
               minK[i] = arr[i];
           }
           return minK;
       }
   };
   ```

   