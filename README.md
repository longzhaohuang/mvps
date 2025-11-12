# mvps
The offical github repo for the paper 

"MVPS: Multi-View Adaptive Prompt Synergy for Zero-shot Anomaly Detection" (ICME 2025 ORAL) 

"MVPS++ :  Local-Aware Multi-View Prompt Synergy For Zero-Shot Anomaly Detection" (Under Review).

## Setting 
### Batch Size and Learning Rate ....
Noting that the setting of our experiment is using BatchSize 4 and learning rate is 1e-4.  Training Epoch is 2.

The setting of experiment is important to the result. 

Methodology framework is provided in this repository, and the complete code will be made available in the future.

### Data Enchancement and Loss Function
We implement our method based on AdaCLIP Github repo. 

Firstly, we do not use data enhancement implement by AdaCLIP. We rotate/flip/resize single image four time to combine a new image instead of selecting four random image. [Very Important]
Secondly, we change the loss function implement detail on AdaCLIP.

### Testing Process
We use cupy to accelerate our testing process. And we allocate memory for each small submatrix individually rather than for the entire large matrix at once, in order to avoid blocking. 
