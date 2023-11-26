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

Completed Activities

1. Run PointNet model on existing datasets (fine tune parameters). Current status - Fine tuning of parameters
2. Run DGCNN model on existing datasets (fine tune parameters). Current status - Fine tuning of parameters
3. Run PointNet on Toronto 3D datasets and capture the metrics. Status - Not started
4. Run DGCNN model on Toronto 3D datasets and capture the metrics. Status - Not started
5. Run SuperPoint Transformer model on Toronto 3D datasets and capture the metrics. Status: Debugging Toronto dataset.
6. Update error functions (based on Hinge loss) in each of the models and re capture the metrics
7. Compare the results and present the results

Pending Activities
1. Fine tune PointNet for Toronto3D and other datasets
2. Fine tune DGCN for Toronto3D and other datasets
3. Fine tune SuperPoint transfomer model
4. Update the models with triplet loss function.
5. Compare results.

   
### Datasets
A Point in space is generally characterized by three parameters, the X, Y, and Z coordinates. A large collection of these points is termed Point Cloud. These are particularly useful for object classification and detection in 3D, which is then utilized for autonomous driving systems. This data is usually captured using 3D scanners or LiDAR technology (Light Detection And Ranging). In practice, each data point consists of various parameters as well apart from coordinates.
XYZ Coordinates
RGB color values
Intensity
GPS time
Scan angle details
There are multiple widely-used point cloud datasets available. Some are them are:
- Semantic KITTI - http://www.semantic-kitti.org/dataset.html
- Paris-Lille-3D - https://npm3d.fr/paris-lille-3d 
- Toronto-3D - https://github.com/WeikaiTan/Toronto-3D 

### Implementation Details
#### 5.1 Activities completed till date
1. Setup of PointNet algorithm and execution on standard (Kitti) dataset.
2. Setup of DGCN (Dynamic Graph Convolution Network) algorithm and execution on standard (Kitti) dataset.
3. Setup of SuperPoint Transformer model.
4. Toronto 3D dataset preprocessing and transformation to use with the SuperPoint Transformer model.
5. Completed preprocessing of Toronto 3D LO04.ply file. Waiting for the other three file transformation to complete before fine tuning the parameters.
6. Completed the modification to include Triplet loss instead of regular cross-entropy loss in the algorithm.

The detailed implementation changes can be found at 
https://github.com/shivansh1010/FML-project/tree/main

#### 5.2 Activities post Midterm report are
1. Integrate Toronto-3D dataset with DGCNN
2. Integrate Toronto-3D dataset with PointNet
3. Fine tune model parameters to improve accuracies.
4. Present the results

#### 5.3 Challenges Faced
Three of the four files in Toronto-3D files are greater than 1GB. Hence gives CUDA out-of-memory error (on a 16GB GPU machine). We have disabled use of GPU and running on CPU mode.
SuperPoint Transformers uses FRNN module which uses a very old version of CUDA library. Downgrading the CUDA library is a challenge as the server is a shared server. We are looking at CPU only FRNN library
Faced a lot of issues in early days related to library version mismatch. We explored various versions and now have finalized on a “conda” environment that is working.

### References
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
