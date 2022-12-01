# Creating-Camouflage
Combining VAE and GAN techniques to create camouflage in images



We have a dataset of images of nature with a size of 150x150. We then select a 15x150 horizontal strip from the image, which we pass into a Variational Auto-Encoder-esque model, the Designer, which will construct a 15x15 'camo' which tries to blend in with the rest of the strip.
While we have another model, the Lookout, which is given the image strip and attempts to discern where the camoflouge is.

The Designer loss is a combination of reconstruction loss of a section from the strip, distribution loss for the reparametrization, and visibility to the Lookout.
The Lookout loss is the difference between its prediction of the camouflage's location and the actual camouflage location, with a percent of the "true" locations incorrect to prevent the Lookout from becoming too good at finding camouflage and making the Designer's job impossible.

We can highlight some examples of the Designer and Lookout after some training, and provide context for them. (each of the images and camouflage locations were generated randomly)

This is an example of what a strip with good camouflage on it looks like, and the associated prediction:
![Good Camouflage Strip](/images/good_camo_strip.png "Good Camouflage Strip")
![Prediction of Good Camouflage](/images/good_camo_pred.png "Prediction of Good Camouflage")

The prediction and true values are not closely linked, demonstrating the efficacy of the camouflage.


This is an example of what a strip with poor camouflage on it looks like, and the associated prediction:
![Bad Camouflage Strip](/images/bad_camo_strip.png "Bad Camouflage Strip")
![Prediction of Bad Camouflage](/images/bad_camo_pred.png "Prediction of Bad Camouflage")

It difficult for the Designer to create proper camouflage for this strip as the features change drastically through the strip, while the prior strip was more uniform,
meaning the prediction is much closer to the true value.

We can also show full images with camouflage within them to highlight the strengths and weaknesses of the system, which can also be a game of spot the camouflage.

This image is a good example of the Designer making effective camouflage for the environment, as it has consistent features:

![Good Camouflage Image](/images/forest_camo_image.png "Good Camouflage Image")

This image highlights how the Designer will create camouflage that fits the general trends of the strip, which won't always fit in with where the camo is place,
as it is mimicking how the water meets the horizon but is placed in front of land:

![Misleading Camouflage Image](/images/misleading_camo_image.png "Misleading Camouflage Image")


In this image, the Designer has to create camouflage that matches how the mountains meet the sky, but the mountain's height changes along the horizontal strip,
making it difficult to assign a crisp edge in the design:

![Mountain Camouflage Image](/images/okay_camo_image.png "Mountain Camouflage Image")


With hyperparameter tuning and longer training time, the camouflages should improve and become more convincing!
