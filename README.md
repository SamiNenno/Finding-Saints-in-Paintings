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
Their results are depicted in the table below. Note that the classes are not exactly the same as in my version. I used the 11 classes (10 saints + NONE) that occurred the most frequent.:<br/>

![](https://raw.githubusercontent.com/SamiNenno/Finding-Saints-in-Paintings/main/Images/Milani_Results.png)
<br/>
<br/>
For the first take on this task, I did not perform data augmentation but normalized the images. The ViT model that I used was the base model and can be found on [huggingface](https://huggingface.co/google/vit-base-patch16-224-in21k). For hyperparameter tuning, I focused on the learning rate and the batch size. For the epochs, early stopping on the validation error was chosen. In most cases training stopped after 5 epochs. Increasing the learning rate correlated with an increase in epochs. Due to computational costs, the maximum batch size was 64, even though a bigger batch size could possibly improve the results. Because of the strong label imbalance I put a focus on (macro) F1 and precision rather than accuracy
<br/>
| model_name | learning_rate | batch_size | Precision(macro) | Precision(micro) | Recall(macro) | Recall(micro) | F1(macro) | F1(micro) | Accuracy |
|------------|---------------|------------|------------------|------------------|---------------|---------------|-----------|-----------|----------|
| ViT_1      | 5e-05         | 2          | 0.52             | 0.63             | 0.54          | 0.63          | 0.51      | 0.63      | 0.63     |
| ViT_2      | 5e-05         | 8          | 0.58             | 0.65             | 0.53          | 0.65          | 0.55      | 0.65      | 0.65     |
| ViT_3      | 5e-05         | 16         | 0.55             | 0.65             | 0.52          | 0.65          | 0.52      | 0.65      | 0.65     |
| ViT_4      | 3e-05         | 2          | 0.59             | 0.67             | 0.53          | 0.67          | 0.55      | 0.67      | 0.67     |
| ViT_5      | 3e-05         | 8          | **0.61**             | 0.67             | 0.56          | 0.67          | **0.57**      | 0.67      | 0.67     |
| ViT_6      | 3e-05         | 16         | 0.61             | **0.67**             | 0.53          | **0.67**          | 0.55      | **0.67**      |**0.67**     |
| ViT_7      | 3e-05         | 64         | 0.53             | 0.63             | 0.51          | 0.63          | 0.51      | 0.63      | 0.63     |
| ViT_8      | 2e-05         | 2          | 0.6              | 0.67             | 0.55          | 0.67          | 0.56      | 0.67      | 0.67     |
| ViT_9      | 2e-05         | 8          | 0.54             | 0.65             | **0.56**          | 0.65          | 0.54      | 0.65      | 0.65     |
| ViT_10     | 2e-05         | 16         | 0.52             | 0.66             | 0.52          | 0.66          | 0.51      | 0.66      | 0.66     |
| ViT_11     | 2e-05         | 64         | 0.52             | 0.66             | 0.44          | 0.66          | 0.45      | 0.66      | 0.66     |
| ViT_12     | 5e-06         | 2          | 0.52             | 0.66             | 0.48          | 0.66          | 0.49      | 0.66      | 0.66     |
| ViT_13     | 5e-06         | 8          | 0.49             | 0.64             | 0.42          | 0.64          | 0.44      | 0.64      | 0.64     |
| ViT_14     | 5e-06         | 16         | 0.47             | 0.64             | 0.38          | 0.64          | 0.39      | 0.64      | 0.64     |
<br/>
ViT_6 (learning rate: 3e-05, batch size: 16) performed best in the most categories and achieves the highest accuracy (0.67). However, as said before, F1 and recall are more important due to the imbalance in labels. Therefore, ViT_5 (learning rate: 3e-05, batch size: 8) with a macro F1 of 0.57 and ViT_9 (learning rate: 2e-05, batch size: 8) with a macro precision of 0.56 are also important for the error analysis. Overall, however, it can be said that the Vision Transformer approach performed worse than the original fine-tuned CNN by Milani et al. Below you can find the confusion matrix for ViT_5

![ViT_5](https://raw.githubusercontent.com/SamiNenno/Finding-Saints-in-Paintings/main/Images/ViT_tune_corrected_6.png)

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
For training on the altered data set, the hyperparameters of the previously best performing models were chosen:
<br/>

| model_name  | learning_rate | batch_size | Precision(macro) | Precision(micro) | Recall(macro) | Recall(micro) | F1(macro) | F1(micro) | Accuracy |
|-------------|---------------|------------|------------------|------------------|---------------|---------------|-----------|-----------|----------|
| ViT_small_1 | 3e-05         | 8          | **0.61**             | 0.6              | 0.58          | 0.6           | **0.58**      | 0.6       | 0.6      |
| ViT_small_2 | 3e-05         | 16         | 0.55             | 0.61             | **0.61**          | 0.61          | 0.57      | 0.61      | 0.61     |
| ViT_small_3 | 2e-05         | 8          | 0.61             | **0.62**             | 0.56          | **0.62**          | 0.57      | **0.62**      | **0.62**     |

<br/>
As it turns out the increase of F1 (+0.01) is only moderate. For an analysis via class activation maps (CAM), I therefore stuck with ViT_5 which performed best according to F1 (0.57) in the first round and is trained on the entire dataset.<br/><br/>
The first image shows CAM for a painting of Virgin Mary. Since one of the iconographic cues for Mary is that she holds baby Jesus, the result should not only focus on Mary herself but also on the baby in her arms. As one can see the trained model performed well on the given image in this regard.

![](https://raw.githubusercontent.com/SamiNenno/Finding-Saints-in-Paintings/main/Images/MARY_02%2B%2B.png)

Another successful CAM is that of an image of Jerome, who is usually depicted as wearing a red coat.

![](https://raw.githubusercontent.com/SamiNenno/Finding-Saints-in-Paintings/main/Images/JEROME_03.png)

However, the next image displays how the model failed to pick up on the relevant iconographic cues. The model correctly classified this painting as showing Mary Magdalene. Yet, the reason for this classification seems to be the fact that the model depicts a woman. The long hair plus the face caused the classification. However, this applies to many persons in paintings and not exclusively to Mary Magdalene. What is more unique to paintings of her is the skull, which does not seem to have a big impact on the classification.

![](https://raw.githubusercontent.com/SamiNenno/Finding-Saints-in-Paintings/main/Images/MM.png)

Another example for a correctly classified painting but based on the wrong features is this one of Sebastian. As mentioned before, Sebastian should be recognized by being tied to a post and shot with arrows. However, the model seem to have rather picked up mostly on the muscular torso.

![](https://raw.githubusercontent.com/SamiNenno/Finding-Saints-in-Paintings/main/Images/SEBASTIAN_01.png)

Overall, most CAM images did **not** highlight the relevant iconographic features. This is not surprising, given the poor performance of even the best models.
