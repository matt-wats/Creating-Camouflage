# Creating-Camouflage
Combining VAE and GAN techniques to create camouflage in images



We have a dataset of images of nature with a size of 150x150. We then select a 15x150 horizontal strip from the image, which we pass into a Variational Auto-Encoder-esque model, the Designer, which will construct a 15x15 'camo' which tries to blend in with the rest of the strip.
While we have another model, the Lookout, which is given the image strip and attempts to discern where the camoflouge is.

The Designer loss is a combination of reconstruction loss of a section from the strip, distribution loss for the reparametrization, and visibility to the Lookout.
The Lookout loss is the difference between its prediction of the camouflage's location and the actual camouflage location, with a percent of the "true" locations incorrect to prevent the Lookout from becoming too good at finding camouflage and making the Designer's job impossible.

We can highlight some examples of the Designer and Lookout after some training, and provide context for them. (each of the images and camouflage locations were generated randomly)

## Example Images
These are images with a camouflage generated for a given horizontal strip, placed in two different column section, demonstrating the Designer's efficacy at creating 
a design that functions anywhere along the strip:

![1st Multi Camouflage Image](/images/multi_camo_image1.png "1st Multi Camouflage Image")
![2nd Multi Camouflage Image](/images/multi_camo_image2.png "2nd Multi Camouflage Image")

Here the same camouflage is applied at y=100 and x=25 and 125, respectively.

### Examples of Full Images
We can also show full images with camouflage within them to highlight the strengths and weaknesses of the system, which can also be a game of spot the camouflage.

This image is a good example of the Designer making effective camouflage for the environment, as it has consistent features:

![Good Camouflage Image](/images/forest_camo_image.png "Good Camouflage Image")

This image highlights how the Designer will create camouflage that fits the general trends of the strip, which won't always fit in with where the camo is place,
as it is mimicking how the water meets the horizon but is placed in front of land:

![Misleading Camouflage Image](/images/misleading_camo_image.png "Misleading Camouflage Image")


In this image, the Designer has to create camouflage that matches how the mountains meet the sky, but the mountain's height changes along the horizontal strip,
making it difficult to assign a crisp edge in the design:

![Mountain Camouflage Image](/images/okay_camo_image.png "Mountain Camouflage Image")

## What the Training Looks Like
### Training Loss Plots

![Designer Loss](/images/designer_loss.png "Designer Loss")
![Lookout Loss](/images/lookout_loss.png "Lookout Loss")

### Examples of the horizontal strips used in training
#### Good Camouflage Strip
This is an example of what a strip with good camouflage on it looks like, and the associated prediction:
![Good Camouflage Strip](/images/good_camo_strip.png "Good Camouflage Strip")
![Prediction of Good Camouflage](/images/good_camo_pred.png "Prediction of Good Camouflage")

The predicted and true values are not closely linked, demonstrating the efficacy of the camouflage.

#### Bad Camouflage Strip
This is an example of what a strip with poor camouflage on it looks like, and the associated prediction:
![Bad Camouflage Strip](/images/bad_camo_strip.png "Bad Camouflage Strip")
![Prediction of Bad Camouflage](/images/bad_camo_pred.png "Prediction of Bad Camouflage")

It difficult for the Designer to create proper camouflage for this strip as the features change drastically through the strip, while the prior strip was more uniform,
meaning the prediction is much closer to the true value.


## Future

Without changing any of the inner-working, hyperparameter tuning and longer training time, the camouflages should improve and become more convincing!

We could change the target prediction and loss weightings for the Lookout training to prioritize the edges of the camouflage so that the Designer focuses more on 
making the transition from real image is camouflage more seamless.

We could change the section of the image that the camouflage is created for from a strip to a square section, in hopes that the section is more homogenous and the Designer will better account of vertical differences.

The current Designer can also create new camouflages without a reference strip, because of it's Encoder-Decoder model, which is only useful in situation where we know
certain features of the target environment but don't have actual photos. If all we care about is the process of photo to camouflage, the Designer's architecture
can be modified to better suit this purpose.
