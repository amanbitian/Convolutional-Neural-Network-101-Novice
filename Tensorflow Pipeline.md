# 🚚 Loading Data:
![tensorflow pipeline](https://user-images.githubusercontent.com/86042628/155850400-f02bb033-eab8-48fb-b013-93e895aa5b2f.PNG)

![tensorflow pipeline_prefetch](https://user-images.githubusercontent.com/86042628/155850416-0c5cc919-39f5-4015-bc05-bee6fd16037f.PNG)

### Concept of tf.data
The tf.data API enables you to build complex input pipelines from simple, reusable pieces. tf.data also makes it possible to handle large amount of data, reading from different data formats, and perform complex transformation

### What is the Tensorflow Input Pipeline?
The input pipeline is a quick and easy utility provided in tf.dataapi to make complex input pipelines from simple and reusable codes and all in few lines of code. It also allows handling a large amount of data, thus giving low-end machines an advantage in computing them.
It does it by wrapping the data into tf.data.dataset class and performing a series of operations on them called ETL - Extract, Transform, Load.

#### The benefits of using **Tensorflow Input Pipeline** are as follows:
1. ***`Loading data`***: Data is loaded in chunks in tf.data.dataset structure or batches, **allowing huge data to be managed and loaded while retaining memory space**. Also, It allows support for different data formats and types including cloud storage (S3).
2. ***`Data manipulation/ augmentation`***:- In general this is done with Pandas, Numpy -text / tabular/ numerical data, or Keras ImageDataGenerator / OpenCV-image data. But here, it can be performed in the input pipeline itself, **allowing faster and rapid prototyping while coding**. **Filtering**, **mapping**, **resizing**, **cropping** are to name a few.
3. ***`One Liner Finally`***: All this can be done in a single line of code thus saving memory space. Here is a quick snippet of it along with an explanation:
4. ***`Distributed Computing`***: It **supports distributed and parallel computing** for enterprises which is essential for cloud computing and big data.

`Note:-` Tensors are the underlying data structures behind tf.data.dataset, so it's different from NumPy arrays or Pandas dataframe. Don’t get mistaken😉.

### How to optimise pipeline?
There are several ways how tf.data reduce computational overhead which can be easily implemented into your pipeline:

1. Prefetching
2. Parallelising data extraction
3. Parallelising data transformation
4. Caching
5. Vectorised mapping

#### 0. Naive approach
Before we start on these concepts, we will have to first understand how the naive approach works when a model is being trained.

![1_NlsHwaUrEsGII-o4LAVWsQ](https://user-images.githubusercontent.com/86042628/155850492-745a05d3-6abc-49b5-aa97-49fdf6df411f.png)


This diagram shows that a training step includes opening a file, fetching a data entry from the file and then using the data for training. We can see clear inefficiencies here as when our model is training, the input pipeline is idle and when the input pipeline is fetching the data, our model is idle.

tf.data solves this issue by using prefetching .

#### 1. Prefetching
Prefetching solves the inefficiencies from naive approach as it aims to overlap the preprocessing and model execution of the training step. In other words, when the model is executing training step n, the input pipeline will be reading the data for step n+1.

The tf.data API provides the tf.data.Dataset.prefetch transformation. It can be used to decouple the time when data is produced from the time when data is consumed. In particular, the transformation uses a background thread and an internal buffer to prefetch elements from the input dataset ahead of the time they are requested.

![1_PISwE0ow6bjhlNME_Uq6Yw](https://user-images.githubusercontent.com/86042628/155850496-cdcc6b70-8652-4eb8-a36d-8dd71b3ecaca.png)


There is an argument that prefetch transformation requires — the number of elements to prefetch. However, we could simply make use of tf.data.AUTOTUNE— provided by tensorflow, which prompts tf.data runtime to tune the value dynamically at runtime.


#### 2. Parallelising data extraction
There exists computational overhead when raw bytes are loaded into memory when reading data, as it may be necessary to deserialise and decrypt the read data. This overhead exists irrespective of whether data is stored locally or remotely.

![1_Amms5OWLomjOe0MJnnQpJw](https://user-images.githubusercontent.com/86042628/155850505-cc1d7075-0d05-4c7a-a935-31e81dee91d7.png)


![1_JGSJ-Ax35uNHAQ9kQeSdQw](https://user-images.githubusercontent.com/86042628/155850508-724eba8b-139b-47cb-8adb-428f95456194.png)


To deal with this overhead, tf.data provides tf.data.Dataset.interleave transformation to parallelise the data loading step, interleaving the contents of other datasets.

Similarly, this interleave transformation supports tf.data.AUTOTUNE which will again delegate the decision of the level of parallelism during tf.data runtime.

#### 3. Parallelising data transformation
In most scenarios, you will have to do some preprocessing to your dataset before passing it to the model for training. The tf.dataAPI takes care of this by offering the tf.data.Dataset.map transformation — which applies a user-defined function to each element of the input dataset.

Because input elements are independent of one another, the pre-processing can be parallelised across multiple CPU cores.

![1_Sbk_IxtsPuUTCV9yQq1ZBA](https://user-images.githubusercontent.com/86042628/155850522-868f32ba-f36c-45bd-9514-f612adf59005.png)

![1_eKf7t8tUgZFIsmxNvuQA0w](https://user-images.githubusercontent.com/86042628/155850525-60b52def-d7e9-4bea-9595-af5756e62820.png)

To utilise multiple CPU cores, you will have to pass in num_parallel_calls argument to specify the level of parallelism you want. Similarly, the map transformation also supports tf.data.AUTOTUNE which will again delegate the decision of the level of parallelism during tf.data runtime.

#### 4. Caching
![1_Jt3WrFTO22qmKoj-eKO5xA](https://user-images.githubusercontent.com/86042628/155850531-c594fc1b-0797-433c-834c-b67d1bf43c9d.png)

tf.data also have caching abilities with tf.data.Dataset.cache transformation. You can either cache a dataset in memory or in local storage. The rule of thumb will be to cache a small dataset in memory and a large dataset in local storage. This thus saves operation like file opening and data reading from being executed during each epoch — next epochs will reuse the data cached by the cache transformation.

One thing to note — you should cache after preprocessing (especially when these preprocessing functions are computational expensive) and before augmentation, as you would not want to store any randomness from your augmentations.

#### 5. Vectorised mapping
![1_wOa4Zv3S_bpZL4eLtBqH1g](https://user-images.githubusercontent.com/86042628/155850539-ff112633-b8c9-423e-9a7b-3bd8a619c0d9.png)

![1_WwyRAkR305Tt-q0uN0FHqg](https://user-images.githubusercontent.com/86042628/155850542-009c40c8-8e03-4b18-830a-975f711a6588.png)

When using the tf.data.Dataset.map transformation as mentioned previously under ‘parallelising data transformation’, there is certain overhead related to scheduling and executing the user-defined function. Vectorising this user-defined function — have it operate over a batch of inputs at once — and applying the batch transformation before the map transformation helps to improve on this overhead.

#### Conclusion
In summary, you can use prefetch transformation to overlap the work done by pipeline (producer) and the model (consumer), interleave transformation to parallelise data reading, map transformation to parallelise data transformation, cache transformation to cache data in memory or local storage and also vectorising your map transformations with the batch transformation.
As mentioned at the start, one of the worst things to experience is seeing your GPU capacity not fully utilised with the bottleneck on the CPU. With tf.data , you will most probably be happy with your GPU utilisation!
