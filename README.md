# Hand Gesture Recognition

A mini hand gesture recognition project.

## Description

This hand gesture recognition project uses the [SqueezeNet](https://github.com/forresti/SqueezeNet) algorithm to be able to recognize a set of 6 different gestures.

## Getting Started

### Dependencies

* 

### Installing

* Download gesture-model.h5 file

### Executing program

```
$ <cmd>
``` 

## Authors

Contributors names and contact info

Jenner Higgins [@Tora12](https://github.com/Tora12)

## Version History

* 0.1
    * Initial Release

## License

This project is licensed under the MIT License - see the LICENSE.md file for details

## Report

For my final project I wanted to do something relating to my senior capstone design project. The main goal of my capstone project is to develop a non-contact gesture based human machine interface (HMI) that can accurately identify gestures in order to replace basic hospital operations with a contact-less system. The application of a non-contact human interaction system highly desirable in addressing this problem of reducing the risk of hospital acquired infections from contaminated surfaces as well as reduce the amount of time hospitals have to spend cleaning and sanitizing each device. At the time of my proposal for this class, my capstone group was still researching whether to use a Leap Motion controller or an Azure Percept vision controller.  During my portion of the research, I noticed that the OpenCV library was already implemented as part of the architecture in the Azure Percept. Therefore for one, I wanted to grasp a better understanding of what was going on behind the scenes in the Percept and for two, wanted to see if I could create a program that could do just as good if not better than the Percept. If I were to be successful in creating a a program that worked nearly as well as the Percept, then I could save my client from spending hundreds of extra dollars on purchasing the Azure Percept vision development kit. 

A couple of assumptions I made for my project include:
Hospital staff members are likely to be wearing gloves for sanitation reasons
The user is likely to be somewhat close to the device as hospital rooms are not always the most spacious and in order to classify gestures with greater accuracy
With this in mind I modified my approach slightly to address these assumptions I had made. First off, I decided to build my own training data set in order to train my model using mostly training images with my hand wearing a glove. I was also hopeful that this would help differentiate the color of my hand from the white wall background. Secondly, I chose to display a smaller region of interest within the camera’s total field of view in attempt to reduce the possibility of picking up background noise.
I would save these images in a sub-directory called training_images and within the sub-directory create another directory with the label of the image which would contain all the training images for that label. I would try to recognize 6 different gestures: a fist, an L shape, a peace sign, an okay sign, an open hand and no gesture. 

After building a training data set, next I would load all the training data images and their labels and create a convolutional neural network model using SqueezeNet. “SqueezeNet is a CNN architecture which has 50 times less parameters than AlexNet but still maintains AlexNet level accuracy” (Atam and QoChuck, 2021). SqueezeNet does this by utilizing 3 specific strategies in order to trim the majority of parameters:
Replacing the 3x3 filters with 1x1 filters
Decreasing the number of input channels to 3x3 filters
Downsampling later at the network

The 3 main advantages that this creates is that: 
It increases the efficiency of distributed training
Uses less overhead when sending out new models to customers
Makes it feasible to implement in Field-programmable Gate Arrays and embedded deployment
Once we create and train the model we save the model to use it in our real-time gesture recognition part of the program. 

In order to do real-time gesture recognition, we need to differentiate the background from our foreground which is our hand. My approach to this is to use the same region of interest area similar to that we used when capturing training images. I differentiate the background from the foreground by finding the running average of the background for the first 30 frames. This assumes that the first 30 frames will only contain the background when first running the program and will not move while the program is running. From there we use contours and convex hull to detect our hand object within the region of interest and run the image for the frame in our model to try a predict the gesture. 

However, our program/model is nowhere near perfect. For one, my model is trained on a relatively small set of data images (roughly 250 images per gesture). Not only this, the training data images were all taken in a very specific environment with very little noise and therefore struggles to perform well in a new environment (i.e. different background, different lighting, etc.) and is very sensitive to any bit of background noise since the training data was very pristine. Some improvements we could make to address this limitation of our model is to collect more training data images. Collecting more diverse training images as well as slightly modifying the original images and saving those could help prevent the model to be overfitted to our specific environment. We could also data augment our training image data to also help create slight variations to our already existing raw data images. Some other methods that could potentially improve the accuracy and reduce noise is to train our model using threshold images and also predict using the threshold images. And since we only have 5 very distinct gesture that we need to recognize, we could also cross compare the number of convexity defects to our predicted gesture to double check if the predicted gesture matches our expected number of convexity defects for the specific gesture. Lastly, one thing I learned is that even though we have a gesture label for nothing (i.e. no gesture and only background) I found that the model never predicts nothing and always predicts one of the 5 gestures. We could fix this by defaulting to display nothing as long as the prediction accuracy of any one label is below a certain accuracy threshold.   
	
  Atam, A. and QoChuck, B., 2021. SqueezeNet Model [with Architecture]. [online] OpenGenus IQ: Computing Expertise & Legacy. Available at: <https://iq.opengenus.org/squeezenet-model/> [Accessed 3 December 2021].
