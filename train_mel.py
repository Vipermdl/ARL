from tqdm import tqdm
from torch.utils import data
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from dataset2017 import ISICDataset
from models.ARL import arlnet50
from models.resnet import *
from crop_transform import *

RANDOM_SEED = 6666


def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    def train(model, dataloader, criterion, optimizer):
        model.train()
        losses = []
        acc = 0.0
        for index, (images, labels, _) in enumerate(dataloader):
            labels = labels.to(device).unsqueeze(1).float()
            images = images.to(device)
            predictions = model(images)
            loss = criterion(predictions, labels)
            logps = F.logsigmoid(predictions)
            ps_ = torch.exp(logps)
            equals = torch.ge(ps_, 0.5).float() == labels
            acc += equals.sum().item()
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss = sum(losses) / len(losses)
        train_acc = acc / len(dataloader.dataset)
        print(f'\ntrain_Accuracy: {train_acc:.5f}, train_Loss: {train_loss:.5f}')
        return train_acc, train_loss

    def validation(model, dataloader, criterion):
        model.eval()
        with torch.no_grad():
            running_acc = 0.0
            val_losses = []
            for index, (images, labels, _) in enumerate(dataloader, start=1):
                labels = labels.to(device).unsqueeze(1).float()
                images = images.to(device)
                # hogs = hogs.to(device)
                score = []
                for i in range(len(images[0])):
                    ps = model(images[:,i])
                    score.append(ps)
                score = sum(score) / len(score)
                logps = F.logsigmoid(score)
                ps_ = torch.exp(logps)
                loss = criterion(score, labels)
                val_losses.append(loss.item())
                equals = torch.ge(ps_, 0.5).float() == labels
                running_acc += equals.sum().item()
            val_loss = sum(val_losses) / len(val_losses)
            val_acc = running_acc / len(dataloader.dataset)
            print(f'\nval_Accuracy: {val_acc:.5f}, val_Loss: {val_loss:.5f}')
        return val_acc, val_loss

    def save_checkpoint():
        filename = os.path.join(checkpoint_dir, "mel_arlnet50_b32_best_acc.pkl")
        torch.save(model.state_dict(), filename)

    def adjust_learning_rate():
        nonlocal lr
        lr = lr / lr_decay
        return optim.SGD(model.parameters(), lr, weight_decay=weight_decay, momentum=0.9)

    # set the parameters
    data_dir = '/home/wuhao/madongliang/dataset/ISIC2017/'
    # Create the dataloaders
    batch_size = 32
    # the checkpoint dir
    checkpoint_dir = "./checkpoint"

    # the learning rate para
    lr = 1e-4
    lr_decay = 2
    weight_decay = 1e-4

    stage = 0
    start_epoch = 0
    stage_epochs = [30, 30, 30, 10]
    total_epochs = sum(stage_epochs)
    writer_dir = os.path.join(checkpoint_dir, "mel_arlnet50")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if not os.path.exists(writer_dir):
        os.makedirs(writer_dir)

    writer = SummaryWriter(writer_dir)

    train_transforms = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.RandomRotation((-10, 10)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.7079057, 0.59156483, 0.54687315],
                             std=[0.09372108, 0.11136277, 0.12577087])
    ])

    val_transforms = argumentation_val()
    # training dataset
    train_dataset = ISICDataset(path=data_dir, mode="training", crop=None, transform=train_transforms, task="mel")
    val_dataset = ISICDataset(path=data_dir, mode="validation", crop=None, transform=val_transforms, task="mel")

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    # get the model
    model = arlnet50(pretrained=True)

    # the loss function
    criterion = nn.BCEWithLogitsLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    criterion = criterion.to(device)

    # the optimizer
    optimizer = optim.SGD(model.parameters(), lr, weight_decay=weight_decay, momentum=0.9)

    # initialize the accuracy
    acc = 0.0
    for epoch in tqdm(range(start_epoch, total_epochs)):

        train_acc, train_loss = train(model, train_loader, criterion, optimizer)
        val_acc, val_loss = validation(model, val_loader, criterion)
        writer.add_scalar("train acc", train_acc, epoch)
        writer.add_scalar("train loss", train_loss, epoch)
        writer.add_scalar("val accuracy", val_acc, epoch)
        writer.add_scalar("val loss", val_loss, epoch)

        if val_acc > acc or val_acc == acc:
            acc = val_acc
            print("save the checkpoint, the accuracy of validation is {}".format(acc))
            save_checkpoint()

        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), "./checkpoint/mel_arlnet50/mel_arlnet50_b32_epoches_{}.pkl".format(epoch + 1))

        if (epoch + 1) in np.cumsum(stage_epochs)[:-1]:
            stage += 1
            optimizer = adjust_learning_rate()
            print('Step into next stage')

        if (epoch + 1) == 50:
            torch.save(model.state_dict(), "./checkpoint/mel_arlnet50/mel_arlnet50_b32_epoches_{}.pkl".format(epoch + 1))


if __name__ == '__main__':
    main()