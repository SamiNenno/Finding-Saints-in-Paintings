# Finding-Saints-in-Paintings
Fine-Tuning Vision Transformer (ViT) for iconography classification on christian paintings.


In this project a Vision Transformer (**ViT**) is fine-tuned for **iconography classification**. Iconography classification is the task to classify persons or objects in paintings based on their iconographic symbols. An example is classifying _Saint Sebastian_ on a christian painting. Since the look of Sebastian varies strongly between painter and epoch the model does not learn to classify the person directly but to classify based on the iconographic symbols that occur in (almost) every painting of the person. In case of Saint Sebastian, the person himself can look very different on different paintings but he is usually depicted as being tied to a post or tree and shot with arrows (as shown in the images below). Therefore, the model should generalize on a person who is tied to something tree-like and who has arrows stuck in his body. 

![](https://github.com/SamiNenno/Finding-Saints-in-Paintings/blob/main/Images/Sebastian_1.jpg)
![](https://github.com/SamiNenno/Finding-Saints-in-Paintings/blob/main/Images/Sebastian_2.jpg)
![](https://github.com/SamiNenno/Finding-Saints-in-Paintings/blob/main/Images/Sebastian_3.jpg)


This experiment is based on the paper ["A Data Set and a Convolutional Model for Iconography Classification in Paintings"](https://dl.acm.org/doi/10.1145/3458885) by Milani et al. from 2021. The data set can be found [here](http://www.artdl.org/)<br/>
In their paper, Milani et al. create an annotated data set of christian paintings. The unprocessed data set consists of 42.479 images with 20 labels. Each label denotes a saint (one exception is the NONE label). Examples are the Virgin Mary (MARY) or Saint Sebastian (SEBASTIAN). Some of the images have multiple labels because multiple saints are depicted. These images are removed. Furthermore, the data set is heavily imbalanced since there are much more MARY and NONE images. Some labels are so rare that they are removed from the data set. You can see the distribution of labels of the preprocessed data set in the table below. Training, validation, and test set are picked in a stratified manner, so that the label distribution stays the same for each split. The image below depicts the distribution in the train split.<br/>

| Frequency | Label            |
|-----------|------------------|
| 12833     | None             |
| 11893     | MARY             |
| 1186      | PETER            |
| 1174      | JEROME           |
| 1139      | JOHN THE BAPTIST |
| 980       | FRANCIS          |
| 907       | MARY MAGDALENE   |
| 681       | CATHERINE        |
| 560       | SEBASTIAN        |
| 523       | PAUL             |
| 502       | JOSEPH           |
![](https://github.com/SamiNenno/Finding-Saints-in-Paintings/blob/main/Images/train_Distribution_original.png)
<br/>
The original paper used a Fully Convolutional ResNet50 network pre-trained on ImageNet. For preprocessing, they:<br/>
- Added padding, so that the images have fixed square size<br/>
- Normalized images<br/>
- Augmented the data by horizontal flip<br/>
<br/>
Their results are depicted in the table below:<br/>

![](https://raw.githubusercontent.com/SamiNenno/Finding-Saints-in-Paintings/main/Images/Milani_Results.png)
<br/>
<br/>
For the first take on this task, I did not perform data augmentation but normalized the images. Note that the classes are not exactly the same as in Milani et al. I used the 11 classes (10 saints + NONE) that occurred the most frequent. Due to the strong label imbalance I put a focus on (macro) F1 and precision rather than accuracy.<br/>
The ViT model that I used can be found on [huggingface](https://huggingface.co/google/vit-base-patch16-224-in21k). For hyperparameter tuning, I focused on the learning rate and the batch size. Due to computational costs, the maximum batch size was 64, even though a bigger batch size could possibly improve the results.
<br/>
|    | model_name  | learning_rate | batch_size | Precision(macro) | Precision(micro) | Precision(weighted) | Recall(macro) | Recall(micro) | Recall(weighted) | F1(macro) | F1(micro) | F1(weighted) | Accuracy |
|----|-------------|---------------|------------|------------------|------------------|---------------------|---------------|---------------|------------------|-----------|-----------|--------------|----------|
| 0  | ViT_tune_1  | 5e-05         | 2          | 0.58             | 0.63             | 0.63                | 0.4           | 0.63          | 0.63             | 0.45      | 0.63      | 0.62         | 0.63     |
| 1  | ViT_tune_2  | 0.0005        | 2          | 0.05             | 0.29             | 0.21                | 0.07          | 0.29          | 0.29             | 0.06      | 0.29      | 0.24         | 0.29     |
| 2  | ViT_tune_3  | 0.005         | 2          | 0.04             | 0.39             | 0.16                | 0.09          | 0.39          | 0.39             | 0.05      | 0.39      | 0.22         | 0.39     |
| 3  | ViT_tune_4  | 0.005         | 8          | 0.04             | 0.39             | 0.16                | 0.09          | 0.39          | 0.39             | 0.05      | 0.39      | 0.22         | 0.39     |
| 4  | ViT_tune_5  | 0.005         | 16         | 0.03             | 0.37             | 0.14                | 0.09          | 0.37          | 0.37             | 0.05      | 0.37      | 0.2          | 0.37     |
| 5  | ViT_tune_6  | 0.005         | 64         | 0.03             | 0.37             | 0.14                | 0.09          | 0.37          | 0.37             | 0.05      | 0.37      | 0.2          | 0.37     |
| 6  | ViT_tune_7  | 3e-05         | 2          | 0.55             | 0.65             | 0.66                | 0.53          | 0.65          | 0.65             | 0.54      | 0.65      | 0.65         | 0.65     |
| 7  | ViT_tune_8  | 3e-05         | 8          | 0.61             | 0.68             | 0.68                | 0.54          | 0.68          | 0.68             | **0.57**      | 0.68      | 0.67         | 0.68     |
| 8  | ViT_tune_9  | 3e-05         | 16         | **0.67**             | **0.69**             | **0.71**            | 0.5           | **0.69**         | **0.69**            | 0.55      | **0.69**      | **0.68**       | **0.69**    |
| 9  | ViT_tune_10 | 3e-05         | 64         | 0.6              | 0.67             | 0.67                | 0.46          | 0.67          | 0.67             | 0.5       | 0.67      | 0.66         | 0.67     |
| 10 | ViT_tune_12 | 5e-05         | 8          | 0.58             | 0.64             | 0.66                | 0.48          | 0.64          | 0.64             | 0.5       | 0.64      | 0.64         | 0.64     |
| 11 | ViT_tune_13 | 5e-05         | 16         | 0.52             | 0.65             | 0.66                | 0.57          | 0.65          | 0.65             | 0.53      | 0.65      | 0.65         | 0.65     |
| 12 | ViT_tune_14 | 5e-05         | 32         | 0.59             | 0.64             | 0.68                | 0.52          | 0.64          | 0.64             | 0.54      | 0.64      | 0.64         | 0.64     |
| 13 | ViT_tune_15 | 5e-05         | 64         | 0.56             | 0.68             | 0.68                | 0.54          | 0.68          | 0.68             | 0.55      | 0.68      | 0.68         | 0.68     |
| 14 | ViT_tune_16 | 2e-05         | 8          | 0.55             | 0.65             | 0.66                | 0.55          | 0.65          | 0.65             | 0.54      | 0.65      | 0.65         | 0.65     |
| 15 | ViT_tune_17 | 2e-05         | 16         | 0.45             | 0.62             | 0.65                | **0.57**         | 0.62          | 0.62             | 0.49      | 0.62      | 0.62         | 0.62     |
| 16 | ViT_tune_19 | 8e-05         | 64         | 0.58             | 0.66             | 0.67                | 0.56          | 0.66          | 0.66             | 0.55      | 0.66      | 0.66         | 0.66     |
| 17 | ViT_tune_20 | 7e-05         | 64         | 0.57             | 0.67             | 0.67                | 0.51          | 0.67          | 0.67             | 0.53      | 0.67      | 0.66         | 0.67     |
| 18 | ViT_tune_21 | 6e-05         | 64         | 0.58             | 0.67             | 0.67                | 0.51          | 0.67          | 0.67             | 0.54      | 0.67      | 0.66         | 0.67     |
<br/>
Vit_tune_9 (learning rate: 3e-05, batch size: 16) performed best in the most categories and achieves the highest accuracy (0.69). However, as said before, F1 and precision are more important due to the imbalance in labels. Therefore, Vit_tune_8 (learning rate: 3e-05, batch size: 8) with a macro F1 of 0.57 and Vit_tune_17 (learning rate: 2e-05, batch size: 16) with a macro precision of 0.57 are also important for the error analysis. Overall, however, it can be said that the Vision Transformer approach performed worse than the original fine-tuned CNN by Milani et al.
<br/>
The poor results can be (partly) explained by the imbalance of classes. Therefore, the experiment was performed a second time on an altered data set. The classes with the highest frequency (MARY and NONE), were randomly shrunk to half of their size. The other classes were augmented by a horizontal flip, such that their amount doubled. Below the illustration of the altered data set:
<br/>
<br/>

| Frequency | Label            |
|-----------|------------------|
| 6431      | None             |
| 5932      | MARY             |
| 2135      | PETER            |
| 2113      | JEROME           |
| 2082      | JOHN THE BAPTIST |
| 1764      | FRANCIS          |
| 1634      | MARY MAGDALENE   |
| 1219      | CATHERINE        |
| 1008      | SEBASTIAN        |
| 942       | PAUL             |
| 904       | JOSEPH           |
![](https://raw.githubusercontent.com/SamiNenno/Finding-Saints-in-Paintings/main/Images/train_Distribution_reduced.png)
<br/>
<br/>
For training on the altered data set, the hyperparameters of the previously best performing models were chosen, the results can be found below:
<br/>

| model_name     | learning_rate | batch_size | Precision(macro) | Precision(micro) | Precision(weighted) | Recall(macro) | Recall(micro) | Recall(weighted) | F1(macro) | F1(micro) | F1(weighted) | Accuracy |
|----------------|---------------|------------|------------------|------------------|---------------------|---------------|---------------|------------------|-----------|-----------|--------------|----------|
| ViT_tune_Red_1 | 3e-05           | 8          | 0.58             | 0.62             | **0.64**                | **0.6**           | 0.62          | 0.62             | **0.59**      | 0.62      | 0.62         | 0.62     |
| ViT_tune_Red_2 | 3e-05          | 16         | 0.52             | 0.58             | 0.61                | 0.6           | 0.58          | 0.58             | 0.55      | 0.58      | 0.58         | 0.58     |
| ViT_tune_Red_3 | 2e-05           | 8          | 0.57             | 0.59             | 0.61                | 0.57          | 0.59          | 0.59             | 0.56      | 0.59      | 0.59         | 0.59     |
| ViT_tune_Red_4 | 2e-05           | 16         | **0.6**             | **0.62**             | 0.63                | 0.57          | **0.62**          | **0.62**             | 0.58      | **0.62**      | **0.62**         | **0.62**     |

<br/>
