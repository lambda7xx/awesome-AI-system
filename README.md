# Awesome AI System

## Contents

- [Papers](#papers)
  - [Training](#training)
  - [Serving](#Serving)
  - [MoE](#MoE)
  - [Schedule](#schedule)
  - [Optimization](#optimzation)
  - [GNN](#GNN)
- [Contribute](#contribute)

## Papers

### Training
- [Bamboo](https://github.com/MachineLearningSystem/bamboo) NSDI'23 
- [TopoOpt: Optimizing the Network Topology for Distributed DNN Training](https://github.com/MachineLearningSystem/TopoOpt) NSDI'23 
- [Optimus-CC: Efficient Large NLP Model Training with 3D Parallelism Aware Communication Compression](https://github.com/MachineLearningSystem/Optimus-CC) ASPLOS'23

- [Slapo: A Schedule Language for Progressive Optimization of Large Deep Learning Model Training ](https://github.com/MachineLearningSystem/slapo)  arxiv 
- [Zeus: Understanding and Optimizing {GPU} Energy Consumption of {DNN} Training](https://github.com/MachineLearningSystem/Zeus) NSDI'23

- [ModelKeeper: Accelerating DNN Training via Automated Training Warmup](https://github.com/MachineLearningSystem/ModelKeeper) NSDI'23

- [HET: Scaling out Huge Embedding Model Training via Cache-enabled Distributed Framework](https://github.com/MachineLearningSystem/Hetu) VLDB'22

-   [FastMoE: A Fast Mixture-of-Expert Training System](https://github.com/MachineLearningSystem/fastmoe)  arXiv preprint arXiv:2103.13262

- [λDNN: Achieving Predictable Distributed DNN Training with Serverless Architectures](https://github.com/MachineLearningSystem/lambdadnn) TC'21

- [HET: Scaling out Huge Embedding Model Training via Cache-enabled Distributed Framework.](https://github.com/PKU-DAIR/Hetu) VLDB'22

- [STRONGHOLD: Fast and Affordable Billion-scale Deep Learning Model Training](https://github.com/MachineLearningSystem/sc22-ae-big_model) SC'22 
- [AMP: Automatically Finding Model Parallel Strategies with Heterogeneity Awareness](https://github.com/MachineLearningSystem/AMP) NeurIPS '22

- [Whale: Efficient Giant Model Training over Heterogeneous {GPUs}](https://github.com/MachineLearningSystem/EasyParallelLibrary) ATC'22

- [Unity: Accelerating DNN Training Through Joint Optimization of Algebraic Transformations and Parallelization](https://github.com/flexflow/FlexFlow) OSDI'22 

- [NASPipe: High Performance and Reproducible Pipeline Parallel Supernet Training via Causal Synchronous Parallelism](https://github.com/MachineLearningSystem/naspipe) ASPLOS'22
- [Out-of-order backprop: an effective scheduling technique for deep learning](https://github.com/MachineLearningSystem/ooo-backprop) Eurosys'22 

- [Varuna: Scalable, Low-cost Training of Massive Deep Learning Models](https://github.com/MachineLearningSystem/varuna) Eurosys'22 

- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM.git) SC'21

- [Chimera: efficiently training large-scale neural networks with bidirectional pipelines](https://github.com/MachineLearningSystem/Chimera) SC'21 

- [Piper: Multidimensional Planner for DNN Parallelization](https://github.com/MachineLearningSystem/piper) NeurIPS'21

- [Efficient Sparse Collective Communication and its application to Accelerate Distributed Deep Learning](https://github.com/MachineLearningSystem/omnireduce.git) SIGCOMM'21

- [PipeTransformer: Automated Elastic Pipelining for Distributed Training of Large-scale Models](https://github.com/MachineLearningSystem/PipeTransformer.git) ICML'21

- [DAPPLE: An Efficient Pipelined Data Parallel Approach for Large Models Training](https://github.com/MachineLearningSystem/DAPPLE) PPOPP'21

- [TeraPipe:Large-Scale Language Modeling with Pipeline Parallelism](https://github.com/MachineLearningSystem/terapipe) ICML'21 

- [PipeSwitch: Fast Pipelined Context Switching for Deep Learning Applications](https://github.com/MachineLearningSystem/PipeSwitch) OSDI'20 

- [KungFu](https://github.com/MachineLearningSystem/KungFu) OSDI'20

- [A Unified Architecture for Accelerating Distributed DNN Training in Heterogeneous GPU/CPU Clusters](https://github.com/bytedance/byteps) OSDI'20

- [PipeDream: Pipeline Parallelism for DNN Training](https://github.com/MachineLearningSystem/pipedream.git) SOSP'19

- [GeePS: Scalable Deep Learning on Distributed GPUs with a GPU-Specialized Parameter Server](https://github.com/MachineLearningSystem/geeps) Eurosys'16

 

- [awesome distributed deep learning](https://github.com/MachineLearningSystem/Awesome-Distributed-Deep-Learning.git)


### Serving
- [Fast and Efficient Model Serving Using Multi-GPUs with Direct-Host-Access](https://github.com/MachineLearningSystem/DeepPlan.git) Eurosys'23
- [Hidet: Task-Mapping Programming Paradigm for Deep Learning Tensor Programs.](https://github.com/MachineLearningSystem/hidet)

- [High-throughput Generative Inference of Large Language Model
with a Single GPU](https://github.com/MachineLearningSystem/FlexGen) 
- [VELTAIR: Towards High-Performance Multi-Tenant Deep Learning Serving via Adaptive Compilation and Scheduling](https://github.com/MachineLearningSystem/VELTAIR_ASPLOS22) ASPLOS'22

- [DVABatch: Diversity-aware Multi-Entry Multi-Exit Batching for Efficient Processing of DNN Services on GPUs](https://github.com/MachineLearningSystem/DVABatch)  ATC'22 

- [Cocktail: A Multidimensional Optimization for Model Serving in Cloud](https://github.com/MachineLearningSystem/cocktail) NSDI'22



- [Serving Heterogeneous Machine Learning Models on Multi-GPU Servers with Spatio-Temporal Sharing](https://github.com/MachineLearningSystem/glet) ATC'22

- [RIBBON: cost-effective and qos-aware deep learning model inference using a diverse pool of cloud computing instances](https://github.com/MachineLearningSystem/SC21_Ribbon) SC'21

- [ INFaaS: Automated Model-less Inference Serving](https://github.com/MachineLearningSystem/INFaaS.git) ATC'21

- [Abacus](https://github.com/MachineLearningSystem/Abacus) SC'21
- [Serving DNNs like Clockwork: Performance Predictability from the Bottom Up](https://github.com/MachineLearningSystem/clockwork) OSDI'20



- [SWARM Parallelism: Training Large Models Can Be Surprisingly Communication-Efficient](https://github.com/MachineLearningSystem/swarm)

- [Merak: An Efficient Distributed DNN Training Framework with Automated 3D Parallelism for Giant Foundation Models](https://github.com/MachineLearningSystem/Merak) 

- [Exploiting Cloud Services for Cost-Effective, SLO-Aware Machine Learning Inference Serving](https://github.com/MachineLearningSystem/MArk-Project) ATC'19 


- [Nexus: a GPU cluster engine for accelerating DNN-based video analysis](https://github.com/MachineLearningSystem/nexus) SOSP'19 

- [Clipper:A low-latency prediction-serving system](https://github.com/ucbrise/clipper) NSDI'17


### MoE
- [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://github.com/stanford-futuredata/megablocks) MLSYS'23 
- [Tutel: Adaptive Mixture-of-Experts at Scale](https://github.com/MachineLearningSystem/tutel-MOE) MLSYS'23

- [FastMoE: A Fast Mixture-of-Expert Training System](https://github.com/MachineLearningSystem/fastmoe-thu) PPOPP'23

- [awesome MoE](https://github.com/MachineLearningSystem/awesome-mixture-of-experts)

- [MoE Paper](https://github.com/MachineLearningSystem/Awesome-Mixture-of-Experts-Papers)

- [AutoMoE: Neural Architecture Search for Efficient Sparsely Activated Transformers](https://github.com/MachineLearningSystem/AutoMoE) 

- [awsome parallelism](https://github.com/MachineLearningSystem/awesome-Auto-Parallelism)

### Schedule
- [ElasticFlow: An Elastic Serverless Training Platform for Distributed Deep Learning](https://github.com/MachineLearningSystem/ElasticFlow-ASPLOS23) ASPLOS'23 
- [Lucid: A Non-Intrusive, Scalable and Interpretable Scheduler for Deep Learning Training Jobs](https://github.com/MachineLearningSystem/Lucid) ASPLOS'23

- [Shockwave: Fair and Efficient Cluster Scheduling for Dynamic Adaptation in Machine Learning](https://github.com/MachineLearningSystem/shockwave) NSDI'23

- [Multi-Resource Interleaving for Deep Learning Training](https://github.com/MachineLearningSystem/Muri) SIGCOMM'22

- [Synergy : Looking Beyond GPUs for DNN Scheduling on Multi-Tenant Clusters](https://github.com/MachineLearningSystem/synergy.git) OSDI'22

- [Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning](https://github.com/MachineLearningSystem/adaptdl) OSDI'21
- [MISO: Exploiting Multi-Instance GPU Capability on Multi-Tenant GPU Clusters](https://github.com/MachineLearningSystem/socc22-miso) SOCC'22 

- [Chronus: A Novel Deadline-aware Scheduler for Deep Learning Training Jobs](https://github.com/MachineLearningSystem/ChronusArtifact) SOCC'21 

- [Heterogeneity-Aware Cluster Scheduling Policies for Deep Learning Workloads](https://github.com/MachineLearningSystem/gavel) OSDI'20

- [Tiresias -- A GPU Cluster Manager for Distributed Deep Learning Training without complete job information](https://github.com/MachineLearningSystem/Tiresias) NSDI'19

- [awesome DL scheduler](https://github.com/MachineLearningSystem/Awesome-DL-Scheduling-Papers.git)



### Optimization
- [Spada: Accelerating Sparse Matrix Multiplication with Adaptive Dataflow](https://github.com/MachineLearningSystem/spada-sim) ASPLOS'23 

- [Hidet: Task Mapping Programming Paradigm for Deep Learning Tensor Programs](https://github.com/MachineLearningSystem/hidet) ASPLOS'23

- [iGniter: Interference-Aware GPU Resource Provisioning for Predictable DNN Inference in the Cloud](https://github.com/MachineLearningSystem/igniter) TPDS'22 

- [CheckFreq: Frequent, Fine-Grained DNN Checkpointing](https://github.com/MachineLearningSystem/CheckFreq) FAST'22

- [Efficient Quantized Sparse Matrix Operations on Tensor Cores](https://github.com/MachineLearningSystem/Magicube) SC'22

- [Characterizing Variability in Large-Scale, Accelerator-Rich Systems](https://github.com/MachineLearningSystem/gpu_variability_sc22_artifact) SC'22 

- [Prediction of the Resource Consumption of Distributed Deep Learning Systems](https://github.com/MachineLearningSystem/Driple) SIGMETRICS'22 
- [Harmony: Overcoming the hurdles of GPU memory capacity to train massive DNN models on commodity servers](https://github.com/MachineLearningSystem/harmony) VLDB'22

- [AI-Enabling Workloads on Large-Scale GPU-Accelerated System: Characterization, Opportunities, and Implications](https://github.com/MachineLearningSystem/HPCA22_SuperCloud) HPCA'22
- [Pets](https://github.com/MachineLearningSystem/PetS-ATC-2022) ATC'22 

- [Breaking the Computation and Communication Abstraction Barrier in Distributed Machine Learning Workloads](https://github.com/parasailteam/coconet.git) ASPLOS'22 

- [Fine-tuning giant neural networks on commodity hardware with automatic pipeline model parallelism](https://github.com/MachineLearningSystem/FTPipe-ATC21-Finetune.git) ATC'21

- [PET: Optimizing Tensor Programs with Partially Equivalent Transformations and Automated Corrections](https://github.com/MachineLearningSystem/pet-osdi21-ae) OSDI'21

- [APNN-TC: Accelerating Arbitrary Precision Neural Networks on Ampere GPU Tensor Cores](https://github.com/MachineLearningSystem/APNN-TC) SC'21
- [https://github.com/MachineLearningSystem/iGUARD](https://github.com/MachineLearningSystem/iGUARD.git) SOSP'21

- [Fluid: Resource-Aware Hyperparameter Tuning Engine](https://github.com/MachineLearningSystem/Fluid) MLSYS'21
- [Baechi: Fast Device Placement on Machine Learning Graphs ](https://github.com/MachineLearningSystem/baechi) SOCC'20 

- [Accpar: Tensor partitioning for heterogeneous deep learning accelerators](https://github.com/MachineLearningSystem/AccPar) HPCA'20 

- [Dynamic Parameter Allocation in Parameter Servers](https://github.com/MachineLearningSystem/AdaPS) VLDB'20 

- [Data Movement Is All You Need: A Case Study on Optimizing Transformers](https://github.com/MachineLearningSystem/substation) 

### GNN
- [COGNN](https://github.com/MachineLearningSystem/CoGNN_info_for_SC22.git) SC'22
- [TC-GNN: Accelerating Sparse Graph Neural Network Computation Via Dense Tensor Core on GPUs](https://github.com/MachineLearningSystem/TCGNN-Pytorch)
- [GNNAdvisor: An Efficient Runtime System for GNN Acceleration on GPUs](https://github.com/MachineLearningSystem/OSDI21_AE-GNN) OSDI'21

- [Marius: Learning Massive Graph Embeddings on a Single Machine](https://github.com/MachineLearningSystem/marius) OSDI'21

- [Dorylus: Affordable, Scalable, and Accurate GNN Training with Distributed CPU Servers and Serverless Threads](https://github.com/MachineLearningSystem/dorylus) OSDI'21 

- [BNS-GCN: Efficient Full-Graph Training of Graph Convolutional Networks with Partition-Parallelism and Random Boundary Node Sampling](https://github.com/MachineLearningSystem/BNS-GCN) MLSYS'22 

- [Accelerating Large Scale Real-Time GNN Inference Using Channel Pruning](https://github.com/MachineLearningSystem/GCNP) VLDB'21 
- [Reducing Communication in Graph Neural Network Training](https://github.com/MachineLearningSystem/CAGNET) SC'20 

- [awesome GNN](https://github.com/chwan1016/awesome-gnn-systems)
## Contribute
We encourage all contributions to this repository. Open an [issue](https://github.com/lambda7xx/awesome-AI-system/issues) or send a [pull request](https://github.com/lambda7xx/awesome-AI-system/pulls).
