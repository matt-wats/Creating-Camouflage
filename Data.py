import torch

def get_data(device):

    train_images = torch.load("train_data/images.pt").to(device)
    test_images = torch.load("test_data/images.pt").to(device)

    # only include images of nature
    train_images = train_images[2191:11652]
    test_images = test_images[437:2499]

    return train_images, test_images




def get_batch(device, data, i, batch_size, r=None):

    section_size = 15
    n, c, _, w = data.size()

    if r is None:
        r = torch.arange(0,n)

    start = i*batch_size
    end = min(n, start+batch_size)

    if start >= end:
        print("ERROR: batch size is non-positive")

    row, col = torch.randint(0,150-section_size+1, size=(2,1), device=device)

    sections = data[r[start:end], :, row:row+section_size, col:col+section_size]
    strips = data[r[start:end], :, row:row+section_size, :]

    return sections, strips



def apply_camo(device, camos, strips):

    section_size = 15

    n = camos.size(0)

    coords = torch.randint(0,150-section_size+1, size=(1,n), device=device)[0]

    camo_strips = strips.detach().clone()
    locs = torch.zeros(size=(n, 150), device=device)
    for i in range(n):
        camo_strips[i, :, :, coords[i]:coords[i]+section_size] = camos[i]
        locs[i, coords[i]:coords[i]+section_size] = 1 / section_size

    return camo_strips, locs



def create_camo_images(device, designer, data, batch_size, r=None):
    section_size = 15
    n, c, _, w = data.size()

    if r is None:
        r = torch.arange(0,n)


    row, col = torch.randint(0,150-section_size+1, size=(2,1), device=device)

    strips = data[r[:batch_size], :, row:row+section_size, :]

    camos,_,_ = designer(strips)

    camo_images = data[r].detach().clone()
    for i in range(batch_size):
        r = row[i]
        c = col[i]
        s = section_size
        camo_images[i, :, r:r+s, c:c+s] = camos[i]


    return camo_images
