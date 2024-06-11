import torch
from torchvision.utils import save_image

from global_name_space import ARGS

def train(dataloader, optimizer_G, optimizer_D, adversarial_loss, generator, discriminator):
    for epoch in range(ARGS.epochs):
        for i, (imgs, _) in enumerate(dataloader):
        # for i, ( (64,1,28,28), _) in enumerate(dataloader):
            
            # Configure input
            real_imgs = imgs.type(ARGS.tensor).to(ARGS.device)
            # (64,1,28,28) = (64,1,28,28).type(ARGS.tensor).to(ARGS.device)

            # Adversarial ground truths
            valid_label = ARGS.tensor(imgs.size(0), 1).fill_(1.0).to(ARGS.device)
            # (64,1) = ARGS.tensor(imgs.size(0), 1).fill_(1.0).to(ARGS.device)
             
            fake_label = ARGS.tensor(imgs.size(0), 1).fill_(0.0).to(ARGS.device) 
            # (64,1) = ARGS.tensor(imgs.size(0), 1).fill_(0.0).to(ARGS.device)

            # -----------------
            #  Train Generator
            # -----------------

            # Sample noise as generator input
            z = torch.normal(mean=0.,std=1.,size=(imgs.shape[0],ARGS.latent_dim)).type(ARGS.tensor).to(ARGS.device)
            # (64,100) = torch.normal(mean=0.,std=1.,size=(imgs.shape[0],ARGS.latent_dim)).type(ARGS.tensor).to(ARGS.device)

            # Generate a batch of images
            gen_imgs = generator(z)
            # (64,1,28,28) = generator( (64,100) ) 

            # Loss measures generator's ability to fool the discriminator
            # This line makes the parameters of both generator and discriminator updated.
            # Homework. Modify the code so that 
            # the parameters of only generator excluding discriminator are updated.
            g_loss = adversarial_loss(discriminator(gen_imgs), valid_label)
            # () = adversarial_loss( (64,1), (64,1) ) 

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid_label)
            # () = adversarial_loss( (64,1), (64,1) )

            # In the below code, gen_imgs.detach() is crucial for the training discriminator. 
            # gen_imgs instead of gen_imgs.detach() will not work properly.
            # Why? Explain.
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake_label)
            # () = adversarial_loss( (64,1), (64,1) )

            d_loss = (real_loss + fake_loss) / 2
            # () = ( () + () ) / 2

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, ARGS.epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(dataloader) + i
            if batches_done % ARGS.sample_interval == 0:
                save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)