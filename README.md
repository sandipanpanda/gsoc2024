<div>
  <p align="center">
    <img width="40%" height="40%" src="/assets/gsoc.svg"/> <img width="20%" height="20%" src="/assets/kubeflow-horizontal.svg"/>
  </p>
</div>

# Google Summer of Code 2024: Integrating JAX with Kubeflow Training Operator for Distributed Training on Kubernetes

- **Contributor**: [Sandipan Panda](https://github.com/sandipanpanda)  
- **Mentors**: [Andrey Velichkevich](https://github.com/andreyvelich), [Yuki Iwai](https://github.com/tenzen-y), [Yuan Tang](https://github.com/terrytangyuan), [Shravan Achar](https://github.com/shravan-achar)  
- **Organization**: [Kubeflow](https://www.kubeflow.org/)
- **Project**: [Integrate JAX with Kubeflow Training Operator](https://summerofcode.withgoogle.com/programs/2024/projects/c29HXzpc)

## Introduction

During the Google Summer of Code (GSoC) 2024, I had the incredible opportunity to contribute to the [Kubeflow](https://www.kubeflow.org/) open-source project by working on the integration of [JAX](https://jax.readthedocs.io/en/latest/) with the [Kubeflow Training Operator](https://github.com/kubeflow/training-operator). The goal of this project was to provide a seamless and efficient way to run distributed computations on CPU using the JAX framework on Kubernetes. Throughout the summer, I collaborated with my mentors, [Andrey Velichkevich](https://github.com/andreyvelich), [Yuki Iwai](https://github.com/tenzen-y), [Yuan Tang](https://github.com/terrytangyuan), and [Shravan Achar](https://github.com/shravan-achar) to build out this feature by extending the Training Operator.

This blog post provides an overview of the project goals, key challenges, solutions implemented, and lessons learned during my journey.

## Project Overview

JAX, a powerful ML framework developed by Google, is highly valued for its flexibility and performance in large-scale distributed computations, especially with its native support for automatic differentiation and hardware accelerators like GPUs and TPUs. The **Kubeflow Training Operator** is a popular Kubernetes component that allows users to run distributed ML training jobs across various frameworks (such as TensorFlow, PyTorch, and XGBoost). However, until now, it lacked direct support for JAX. 

### Objectives

1. **Create a Custom Resource for JAX (JaxJob):**
   We needed to introduce a new Kubernetes Custom Resource Definition (CRD) for JAX, called `JAXJob`, that would allow users to define distributed JAX training jobs in Kubernetes clusters. This was crucial for enabling the integration of JAX into the Training Operator.

2. **Update the Training Operator Controller:**
   The Training Operator controller had to be updated to support the new `JAXJob` resource, handling the creation, scheduling, and management of distributed JAX training jobs on Kubernetes.

3. **Enhance the Training Operator Python SDK:**
   We aimed to extend the Training Operator Python SDK to provide easy-to-use APIs for data scientists and ML practitioners to define and launch `JAXJob` on Kubernetes, simplifying the process of running distributed JAX jobs.

### Skills and Technology Stack

The project required a strong understanding of the following technologies:
- **Kubernetes**: Building, deploying, and managing distributed workloads using custom resources.
- **Go**: Development of the Kubernetes Training Operator and controller.
- **YAML**: Defining Kubernetes manifests and CRDs.
- **Python**: Integration of JAX with the Python SDK and simplifying job creation workflows.

## Key Contributions

### 1. **Creating the `JAXJob` Custom Resource**

   - The first major task was to define a new **Custom Resource** for JAX jobs, similar to the existing `TFJob`, `PyTorchJob`, and `XGBoostJob`. This required defining a Kubernetes CRD that would describe the specifications for a JAX distributed training job, such as the number of workers, resource allocation, and job configuration.
   
   - The `JAXJob` CRD was designed to be flexible and compatible with other Kubernetes-based workflows. Here’s a basic example of a `JAXJob` manifest:

   ```yaml    
    apiVersion: "kubeflow.org/v1"
    kind: JAXJob
    metadata:
    name: jaxjob-simple
    spec:
    jaxReplicaSpecs:
        Worker:
        replicas: 2
        restartPolicy: OnFailure
        template:
            spec:
            containers:
            - name: jax-worker
                image: sandipanify/jaxgoogle
                command: ["python", "train.py"]
                ports:
                - containerPort: 6666
                imagePullPolicy: Always

   ```

   - With this CRD, data scientists can define JAX-based workloads that are then distributed across multiple nodes in a Kubernetes cluster.

### 2. **Extending the Training Operator Controller**

   - The next step was to **update the Training Operator controller** (written in Go) to recognize and manage `JAXJob` resources. This required adding JAX-specific logic to handle job creation, scheduling, scaling, and monitoring.
      
   - The main controller logic involves watching for `JAXJob` events (create, update, delete) and ensuring that the right resources (e.g., Pods and services) are spun up or down in response to the job’s lifecycle.

   I followed the existing patterns in the Training Operator for other frameworks (such as PyTorch and XGBoost) and adapted them for JAX, ensuring consistency and reusability of the codebase.

### 3. **Enhancing the Training Operator SDK**

To make this new functionality more accessible to users, I extended the Training Operator’s Python SDK. The SDK is widely used by data scientists to interact with Kubernetes resources programmatically, and adding support for JAX was a crucial step toward usability.

This SDK enhancement bridges the gap between data scientists and Kubernetes infrastructure, allowing them to focus on model development rather than cluster management.

### 4. **Testing and Documentation**

Testing was a critical aspect of the project. I implemented both unit and integration tests to ensure that the `JaxJob` CRD and the Training Operator controller functioned correctly under different scenarios, such as node failures, pod restarts, and resource contention.

### Progress and Achievements

By the end of the project, the following milestones were successfully achieved:
- **Creation of enhancemnet proposal** for JAX integration with the Training Operator.
- **Creation of the JAXJob CRD**: Defined a fully functional Custom Resource for JAX jobs, complete with relevant fields and schema.
- **Controller Logic for JAXJob**: Updated the Training Operator controller to manage JAX-specific jobs, ensuring smooth scheduling and execution across distributed nodes.
- **Implement webhook validations for the JAXJob**
- **Basic SDK Integration**: Integrated basic support for `JAXJob` creation in the Python SDK, making it easier for end-users to interact with JAX jobs programmatically.
  
### Pull requests:
- https://github.com/kubeflow/training-operator/pull/2125
- https://github.com/kubeflow/training-operator/pull/2163
- https://github.com/kubeflow/training-operator/pull/2194

### Challenges

1. **Understanding JAX’s Distributed Framework**:
   - One of the initial hurdles was gaining a deep understanding of how JAX handles distributed training on CPU backend as there was lack of enough documentation.

2. **Kubernetes Complexity**:
   - Managing distributed jobs on Kubernetes can be complex, especially when dealing with scaling, fault tolerance, and resource allocation. These challenges were addressed by closely following best practices in Kubernetes CRD design and leveraging the existing infrastructure in the Training Operator.

3. **Controller Design**:
   - Modifying the existing Go-based Training Operator controller to support JAX while ensuring backward compatibility with other frameworks required careful design and testing.

### Future Work

- **Enhanced SDK**: The Python SDK integration is currently basic and requires testing.
- **Testing and Documentation**: Implementing more comprehensive testing across various Kubernetes environments and hardware accelerators is necessary to ensure robustness. Additional documentation and tutorials would also be beneficial to onboard new users to the JAX + Kubeflow ecosystem.

## Lessons Learned

Throughout this project, I gained valuable insights into distributed systems, Kubernetes resource management, and the inner workings of machine learning frameworks like JAX. Some key takeaways include:

- **Kubernetes Deep Dive:** I deepened my understanding of Kubernetes, particularly Custom Resource Definitions (CRDs) and controllers, which are the backbone of extending Kubernetes functionality.
- **Collaboration in Open Source:** Working in a collaborative environment with experienced mentors was one of the highlights of this project. Their feedback and guidance helped me improve not only my technical skills but also my ability to communicate and collaborate effectively.
- **Distributed Training at Scale:** This project gave me a deeper appreciation for the complexities of distributed training and the importance of tools like Kubernetes in managing large-scale machine learning workloads.

### Conclusion

Integrating JAX with the Kubeflow Training Operator has been a challenging but rewarding experience. The project successfully enables distributed training for JAX on Kubernetes, providing an easy-to-use interface for data scientists and machine learning engineers. 

I am grateful to my mentors — [Andrey Velichkevich](https://github.com/andreyvelich), [Yuki Iwai](https://github.com/tenzen-y), [Yuan Tang](https://github.com/terrytangyuan), [Shravan Achar](https://github.com/shravan-achar), and [Johnu George](https://github.com/johnugeorge) — for their support and guidance throughout the summer.

I look forward to seeing how this feature evolves and benefits the Kubeflow community in the future.

