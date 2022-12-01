import torch
import torch.nn as nn


class Designer(nn.Module):

    def __init__(self):
        super().__init__()

        self.camo_size = 15
        image_channels = 3


        # encoder layers
        encoder_filter_size = 3
        encoder_stride = 2
        encoder_filter_num = 8

        encoder_layers_num = 2
        encoder_layers = []
        in_channels = image_channels
        for i in range(encoder_layers_num):
            features = (2**i) * encoder_filter_num
            encoder_layers.append(nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=encoder_filter_size, stride=encoder_stride),
                nn.BatchNorm2d(num_features=features, affine=True),
                nn.ReLU()
            ))
            in_channels = features


        encoder_layers.append(nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=encoder_filter_size, stride=encoder_stride),
        ))


        self.encoder = nn.Sequential(*encoder_layers)

        self.features_size = (2**(encoder_layers_num-1)) * encoder_filter_num

        self.encoder_fc = nn.Sequential(
            nn.Linear(in_features=self.features_size*17, out_features=self.features_size)
        )

        self.fc_mu = nn.Linear(in_features=self.features_size, out_features=self.features_size)
        self.fc_log_var = nn.Linear(in_features=self.features_size, out_features=self.features_size)




        # decoder
        self.decoder_projection = nn.Sequential(nn.Linear(in_features=self.features_size, out_features=(self.camo_size**2)*self.features_size))


        decoder_filter_size = 3
        decoder_filter_num = 64

        decoder_layers_num = 3
        decoder_layers = []
        in_channels = self.features_size
        for i in range(decoder_layers_num-1, -1, -1):
            features = (2**(i-1)) * decoder_filter_num
            if i > 0:
                decoder_layers.append(nn.Sequential(
                    CroppedTransConv(self.camo_size, in_channels=in_channels, out_channels=features, kernel_size=decoder_filter_size),
                    nn.BatchNorm2d(num_features=features, affine=True),
                    nn.ReLU()
                ))
            else:
                decoder_layers.append(nn.Sequential(
                    CroppedTransConv(self.camo_size, in_channels=in_channels, out_channels=image_channels, kernel_size=decoder_filter_size),
                    nn.Sigmoid()
                ))
            in_channels = features

        self.decoder = nn.Sequential(*decoder_layers)


    def forward(self, x):

        mu, log_var = self.encode(x)

        x = self.reparameterize(mu, log_var)

        x = self.decode(x)

        return x, mu, log_var


    def encode(self,x):

        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.encoder_fc(x)

        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)

        return mu, log_var

    
    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        vals = mu + (eps * std)

        return vals


    def decode(self,x):

        x = self.decoder_projection(x)
        x = x.reshape(x.size(0),self.features_size, self.camo_size, self.camo_size)

        x = self.decoder(x)

        return x





class CroppedTransConv(nn.Module):

    def __init__(self, size, in_channels, out_channels, kernel_size):
        super().__init__()


        self.layer = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1)


        self.crop_start = (kernel_size-1) // 2
        self.crop_stop = self.crop_start + size


    def forward(self, x):

        x = self.layer(x)
        x = x[..., self.crop_start:self.crop_stop, self.crop_start:self.crop_stop]

        return x



class Lookout(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        camo_size = 15
        image_channels = 3

        # encoder layers
        filter_size = 3
        stride = 1
        filter_num = 4

        layers_num = 3
        layers = []
        in_channels = image_channels
        for i in range(layers_num):
            features = (2**i) * filter_num
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=filter_size, stride=stride, padding=(0,1)),
                nn.BatchNorm2d(num_features=features, affine=True),
                nn.ReLU()
            ))
            in_channels = features

        
        h = camo_size - layers_num*(filter_size - 1)

        layers.append(nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=(h,1)),
        ))

        self.layers = nn.Sequential(*layers)

    
    def forward(self, x):

        x = self.layers(x)

        x = x.view(-1, 150)

        x = nn.MaxPool1d(kernel_size=7, stride=1, padding=3)(x)
        x = nn.Softmax(dim=1)(x)

        return x