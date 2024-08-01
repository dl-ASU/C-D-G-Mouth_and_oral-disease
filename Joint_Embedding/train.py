from helpful.libraries import *
from helpful.sampling import *
from helpful.losses import *

def train(train_generator, test_generator, model, optimizer, args):
    print("Training Started ....")

    loss_fn = nn.MSELoss()
 
    do = nn.Dropout(0.15)

    for epoch in range(args.epochs):

        # Training Mode
        model.train()

        train_loss = 0
        for train_x, train_y in train_generator:
          pos, neg = sample_contrastive_pairs_SL(train_x, train_y, args.N)
          train_x, pos, neg, train_y = train_x.to(device), pos.to(device), neg.to(device), train_y.to(device)
          shape = train_x.shape

          # Feedforward
          mask = do(torch.ones(train_x.shape)).to(device)
          masked_img = (mask * train_x).to(device)
          emb_actu = model.encode(masked_img)
          emb_pos = model.encode(pos)
          emb_neg = model.encode(neg.reshape(-1, 1, shape[-2], shape[-1])).reshape(-1, args.N, 2)
          y_pred = model.decode(emb_actu)

          # Calculate the loss function and the accuracy
          _std_loss = std_loss(emb_actu, emb_pos)
          _cov_loss = cov_loss(emb_actu, emb_pos)
          loss_contra = contrastive_loss(emb_actu, emb_pos, emb_neg)
          loss_recon = loss_fn(y_pred.view((-1, shape[-2] * shape[-1])), train_x.view((-1, shape[-2] * shape[-1])))

          train_loss += (loss_recon.item() + loss_contra.item() + _std_loss.item() + _cov_loss.item())
          loss = loss_recon + loss_contra + _std_loss + _cov_loss

          # At start of each Epoch
          optimizer.zero_grad()

          # Do the back probagation and update the parameters
          loss.backward()
          optimizer.step()

        train_loss /= len(train_generator)

        # Evaluation mode
        model.eval()

        with torch.inference_mode():
          test_loss = 0

          for test_x, test_y in test_generator:
              pos, neg = sample_contrastive_pairs_SL(test_x, test_y, args.N)

              test_x, test_y = test_x.to(device), test_y.to(device)

              # Feedforward again for the evaluation phase
              emb_actu = model.encode(test_x)
              emb_pos = model.encode(pos)
              emb_neg = model.encode(neg.reshape(-1, 1, 28, 28)).reshape(-1, args.N, 512)
              y_pred_test = model.decode(emb_actu)

              # Calculate the loss for the test dataset
              _std_loss = std_loss(emb_actu, emb_pos)
              _cov_loss = cov_loss(emb_actu, emb_pos)
              loss_recon = loss_fn(y_pred_test.view((-1, 28 * 28)), test_x.view((-1, 28 * 28)))
              loss_contra = contrastive_loss(emb_actu, emb_pos, emb_neg)
              
              test_loss += (loss_recon.item() + loss_contra.item() + _std_loss.item() + _cov_loss.item())

        test_loss /= len(test_generator)

        print(f"Epoch : {epoch + 1} | training_Loss: {train_loss:.4f} | testing_Loss: {test_loss:.4f}")