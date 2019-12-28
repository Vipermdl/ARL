import json
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils import data
from dataset2017 import ISICDataset
from sklearn import metrics
import matplotlib.pyplot as plt
from crop_transform import *
from models.ARL import arlnet50
RANDOM_SEED = 6666


def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # set the parameters
    checkpoint_dir = "/home/wuhao/madongliang/isic2017/checkpoint/mel_arlnet50_b32_best_acc.pkl"
    result_dir = "./result"
    data_dir = '/home/wuhao/madongliang/dataset/ISIC2017/'
    # Create the dataloaders
    batch_size = 1

    y = []
    y_score = []

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    def imshow(y_pre, y_score):
        fpr, tpr, thresholds = metrics.roc_curve(y_pre, y_score)
        auc = metrics.auc(fpr, tpr)
        print(auc)

        plt.plot(fpr, tpr, c='r', lw=2, alpha=0.7, label=u'AUC=%.3f' % auc)
        plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
        plt.xlim((-0.01, 1.02))
        plt.ylim((-0.01, 1.02))
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xlabel('False Positive Rate', fontsize=13)
        plt.ylabel('True Positive Rate', fontsize=13)
        plt.grid(b=True, ls=':')
        plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
        plt.title(u'ROC and AUC for ISBI2017', fontsize=17)
        plt.savefig("2017_mel_arlnet50_e100_b32.png")


    def load_checkpoint(checkpoint_path):

        # Here put the pretrained model that you used (in my case it's densenet161).

        # model = resnet50()
        # # model = danet()
        # # model = resnet50_cbam(pretrained=False)
        # # model = se_resnet50()
        # # model = proposed()
        # # model = models.resnext50_32x4d(pretrained=False)
        # try:
        #     n_ftrs = model.classifier.in_features
        #     model.classifier = classifier(n_ftrs)
        # except AttributeError:
        #     n_ftrs = model.fc.in_features
        #     model.fc = classifier(n_ftrs)
        # model = model.to(device)
        '''
        fn1 = FeatureNet_1()
        fn2 = FeatureNet_2()
        cfn = ClassifierNet(fn1, fn2) 
        model = cfn
        '''
        model = arlnet50(pretrained=True)
        # checkpoint = torch.load(checkpoint_path, map_location='cpu')
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        '''
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:]  # remove module.
            new_state_dict[name] = v
        '''
        model.load_state_dict(checkpoint)  # your checkpoint's key may differ (e.g.'state_dict')
        model.eval()
        return model

    def predict(model, dataloader):
        mel_tn = 0
        mel_fp = 0
        mel_tp = 0
        mel_fn = 0
        model.eval()
        with torch.no_grad():
            for ii, (images, labels, _) in tqdm(enumerate(dataloader, start=1)):
                images = images.to(device)
                scores = []

                for i in range(len(images[0])):
                    pred = model(images[:, i])
                    scores.append(pred)
                scores = sum(scores) / len(scores)
                logps = F.logsigmoid(scores)
                score = torch.exp(logps)
                pre = torch.ge(score, 0.5).float()
                if int(pre) == 0 and int(labels) == 0:
                    mel_tn += 1
                elif int(pre) == 1 and int(labels) == 0:
                    mel_fp += 1
                elif int(pre) == 1 and int(labels) == 1:
                    mel_tp += 1
                elif int(pre) == 0 and int(labels) == 1:
                    mel_fn += 1
                score = score.cpu().numpy().tolist()[0]
                label = labels.cpu().numpy().tolist()[0]
                y.append(label)
                y_score.append(score)
        return mel_tp, mel_tn, mel_fp, mel_fn

    val_transforms = argumentation_val()
    # Validation dataset
    val_dataset = ISICDataset(path=data_dir, mode="testing", crop=None, transform=val_transforms, task="mel")
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")

    model = load_checkpoint(checkpoint_path=checkpoint_dir)

    model.cuda()
    mel_tp, mel_tn, mel_fp, mel_fn = predict(model, val_loader)

    mel_acc = (mel_tp + mel_tn) / (mel_tn + mel_fp + mel_tp + mel_fn)
    mel_sen = mel_tp / (mel_tp + mel_fn)
    mel_spe = mel_tn / (mel_tn + mel_fp)

    y_score = np.array(y_score)
    mel_auc = metrics.roc_auc_score(y, y_score)

    imshow(y, y_score)

    print('mel_Accuracy:', mel_acc)
    print('mel_Sensitive:', mel_sen)
    print('mel_Specificity:', mel_spe)
    print('mel_AUC:', mel_auc)
    with open('result.txt', 'a') as f:
        f.write('\n2017_mel_arlnet50_e100_b32: ' + json.dumps(
            {'mel_Accuracy': mel_acc, 'mel_Sensitive': mel_sen, 'mel_Specificity': mel_spe, 'mel_AUC': mel_auc}))


if __name__ == '__main__':
    main()