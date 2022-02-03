# Finding-Saints-in-Paintings
Fine-Tuning Vision Transformer (ViT) for iconography classification on christian paintings.


In this project a Vision Transformer (**ViT**) is fine-tuned for **iconography classification**. Iconography classification is the task to classify persons or objects in paintings based on their iconographic symbols. An example is classifying _Saint Sebastian_ on a christian painting. Since the look of Sebastian varies strongly between painter and epoch the model does not learn to classify the person directly but to classify based on the iconographic symbols that occur in (almost) every painting of the person. In case of Saint Sebastian, the person himself can look very different on different paintings but he is usually depicted as being tied to a post or tree and shot with arrows (as shown in the images below). Therefore, the model should generalize on a person who is tied to something tree-like and who has arrows stuck in his body. 

![](https://github.com/SamiNenno/Finding-Saints-in-Paintings/blob/main/Images/Sebastian_1.jpg)
![](https://github.com/SamiNenno/Finding-Saints-in-Paintings/blob/main/Images/Sebastian_2.jpg)
![](https://github.com/SamiNenno/Finding-Saints-in-Paintings/blob/main/Images/Sebastian_3.jpg)


This experiment is based on the paper ["A Data Set and a Convolutional Model for Iconography Classification in Paintings"](https://dl.acm.org/doi/10.1145/3458885) by Milani et al. from 2021.<br/>
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
