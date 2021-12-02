# Cat vs. Dog classification via Transfer Learning

<img src="https://github.com/dylanlrrb/cat_dog_classifier/blob/main/assets/app.jpeg?raw=true" alt="app_image" width="500"/>

## About

Interactive app is deployed at [https://cat-dog-classifier.onrender.com/](https://cat-dog-classifier.onrender.com/)

This is a proof of concept that uses transfer learning in a camera based webapp to classify cats and dogs

A convolutional neural net is trained on cat and dog images on top of a [pre-trained vgg16 model](https://neurohive.io/en/popular-networks/vgg16/) for lower level feature detection  

[View the notebook here](https://github.com/dylanlrrb/cat_dog_classifier/blob/main/model/Cat_Dog_Classifier.ipynb)



## Getting Started
  
- [ ] Set up a python environment (>=3.7) and install node
- [ ] Run `yarn setup` to install dependencies and download a model already trained on cat and dog images
- [ ] Run `yarn serve:backend` and navigate to [http://localhost:8080/](http://localhost:8080/)
- [ ] You may download the training, testing, and validatation dataset for running the notebook with the command `yarn get:dataset`
 


Notes:
- The webapp is styled for mobile browsers, it is recommended that you toggle on the device toolbar in the developer tools for the best experiance
- Running `Cat_Dog_Classifier.ipynb` from start to finish will generate a new trained model for use in the classification api.
- Doing so will overwrite the one downloaded from running the setup script. if you need the originally downloaded model again, re-run `yarn setup`
