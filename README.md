# Finding-Saints-in-Paintings
Fine-Tuning Vision Transformer (ViT) for iconography classification on christian paintings.


In this project a Vision Transformer (**ViT**) is fine-tuned for **iconography classification**. Iconography classification is the task to classify persons or objects in paintings based on their iconographic symbols. An example is classifying _Saint Sebastian_ on a christian painting. Since the look of Sebastian varies strongly between painter and epoch the model does not learn to classify the person directly but to classify based on the iconographic symbols that occur in (almost) every painting of the person. In case of Saint Sebastian, the person himself can look very different on different paintings but he is usually depicted as being tied to a post or tree and shot with arrows (as shown in the images below). Therefore, the model should generalize on a person who is tied to something tree-like and who has arrows stuck in his body. 

![](https://github.com/SamiNenno/Finding-Saints-in-Paintings/blob/main/Images/Sebastian_1.jpg)
![](https://github.com/SamiNenno/Finding-Saints-in-Paintings/blob/main/Images/Sebastian_2.jpg)
![](https://github.com/SamiNenno/Finding-Saints-in-Paintings/blob/main/Images/Sebastian_3.jpg)


This experiment is based on the paper ["A Data Set and a Convolutional Model for Iconography Classification in Paintings"](https://dl.acm.org/doi/10.1145/3458885) by Milani et al. from 2021.
In their paper, Milani et al. create an annotated data set of christian paintings. In the unprocessed data, there are 20 labels. Each label denotes a saint. Examples are the Virgin Mary (MARY) or Saint Sebastian (SEBASTIAN).
