# Creating-Camouflage
Combining VAE and GAN techniques to create camouflage in images



We have a dataset of images of nature with a size of 150x150. We then select a 15x150 horizontal strip from the image, which we pass into a Variational Auto-Encoder-esque model, the Designer, which will construct a 15x15 'camo' which tries to blend in with the rest of the strip.
While we have another model, the Lookout, which is given the image strip and attempts to discern where the camoflouge is.

The Designer loss is a combination of reconstruction loss of a section from the strip, distribution loss for the reparametrization, and visibility to the Lookout.
The Lookout loss is the difference between its prediction of the camouflage's location and the actual camouflage location, with a percent of the "true" locations incorrect to prevent the Lookout from becoming too good at finding camouflage and making the Designer's job impossible.

We can highlight some examples of the Designer and Lookout after some training, and provide context for them.

This is an example of what a strip with good camouflage on it looks like, and the associated prediction:
![Good Camouflage Strip](/images/good_camo_strip.png "Good Camouflage Strip")
![Prediction of Good Camouflage](/images/good_camo_pred.png "Prediction of Good Camouflage")


This is an example of what a strip with poor camouflage on it looks like, and the associated prediction:
![Bad Camouflage Strip](/images/bad_camo_strip.png "Bad Camouflage Strip")
![Prediction of Bad Camouflage](/images/bad_camo_pred.png "Prediction of Bad Camouflage")

It difficult for the Designer to create proper camouflage for this strip as the features change drastically through the strip, while the prior strip was more uniform.

Here are full images with camouflage within them to highlight the strengths and weaknesses of the system.
