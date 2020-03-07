# CS390-Lab1
What I did to improve my CNN was I added more nodes to each layer than what was in the example 
from the slides. In addition, I added more layers in general, basing it more off of VGG 16. 
I also added a dropout after every pooling to try to combat overfitting. I also added a 
padding to all of my Conv2D layers to pad the outside of the image with zeros to improve the 
max pooling, improving the overall accuracy. I also implemented batches to help improve the accuracy.