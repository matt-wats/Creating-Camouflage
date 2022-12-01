# Creating-Camouflage
Combining VAE and GAN techniques to create camouflage in images



We have a dataset of images of nature with a size of 150x150. We then select a 15x150 horizontal strip from the image, which we pass into a Variational Auto-Encoder-esque model, the Designer, which will construct a 15x15 'camo' which tries to blend in with the rest of the strip.
While we have another model, the Lookout, which is given the image strip and attempts to discern where the camoflouge is.

The Designer loss is a combination of reconstruction loss of a section from the strip, distribution loss for the reparametrization, and visibility to the Lookout.
The Lookout loss is the difference between its prediction of the camouflage's location and the actual camouflage location, with a percent of the "true" locations incorrect to prevent the Lookout from becoming too good at finding camouflage and making the Designer's job impossible.
