Midterm report

    An enhanced and well-defined problem statement, taking into account the received feedback.
    Description of the proposed solution approach.
    A code survey where you include links to the relevant codebases that you refer to while implementing the solution
    Datasets
    Implementation details.
    Preliminary results (if any).
    A roadmap for the remainder of the semester

# FML-project

This repository belongs to the project of CS725-2023, Foundations of Machine Learning course. 

### Topic: Point Cloud Object Detection

World health organization has estimated an average of 1.3 million road deaths every year. Road marking extraction is emerging as an important remote sensing application to meet United Nations goal to reduce road injuries by half by 2030. One of the ways to reduce Highway injuries is to ensure Road marking, Lighting Poles and Guard Rails are in place and in proper condition. The process of analysis of these highway objects is a manual, tedious and time-consuming process. With advancements in data capture technologies, cost of 3D point cloud data capture have drastically reduced over the last decade. Today point cloud and photogrammetry data capture methods are widely used and process to identify road assets and its conditions to monitor and maintain road safety. 

Machine learning algorithms for point cloud has evolved since 2017. PointNet was the first algorithms to use 3D based convolution in 2019. Since then, different methods of 3D based neural network algorithms have evolved. Different methods of convolution for 3D point cloud include voxelization, graph network methods and point cloud transformers each with its own advantages and disadvantages. 

In this project, we would look at exploring three different neural network models (Convolution based, Graph Based and Transformer based) for classification of Fence, Road Marking and Poles (if possible). We would compare the performance of this model using Hinge Loss and see if the model performce better (giving limited labelled datasets). Below are the activities that would be performed as part of this project.

1. Run PointNet model on existing datasets (fine tune parameters). Current status - Fine tuning of parameters
2. Run DGCNN model on existing datasets (fine tune parameters). Current status - Fine tuning of parameters
3. Run PointNet on Toronto 3D datasets and capture the metrics. Status - Not started
4. Run DGCNN model on Toronto 3D datasets and capture the metrics. Status - Not started
5. Run SuperPoint Transformer model on Toronto 3D datasets and capture the metrics. Status: Debugging Toronto dataset.
6. Update error functions (based on Hinge loss) in each of the models and re capture the metrics
7. Compare the results and present the results


### Datasets


### Implementation Details
(Santhosh)

### Preliminary results
(Santhosh)

### References

### Links
- Project Report: https://docs.google.com/document/d/17OKdrrWXopel2GoBX1O-22-uBhzP_fFB9dWt9vmu528/edit
- Discussion Sheet: https://docs.google.com/spreadsheets/d/1oOCtOhW92rF0gF32LmwjMK1lHeleryVZe_BT46rSyU4/edit#gid=0
- Pointnet Slides - PointNet (stanford.edu)
- Pointnet Paper - arxiv.org/pdf/1612.00593.pdf
- Pointnet Github Code - https://github.com/charlesq34/pointnet
- DGCNN Paper - https://arxiv.org/pdf/1801.07829.pdf
- DGCNN Github Code - https://github.com/WangYueFt/dgcnn
- Superpoint Transformer Paper - https://arxiv.org/abs/2306.08045
- Superpoint Transformer Github Code - 
- https://github.com/drprojects/superpoint_transformer/tree/master
- Toronto-3D dataset: https://github.com/WeikaiTan/Toronto-3D
