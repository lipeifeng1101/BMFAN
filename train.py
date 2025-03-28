import os
import argparse
#from torchsummary import summary
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision.models as models
from my_dataset import MyDataSet
#from model import mobile_vit_xx_small as create_model
from utils import read_split_data, train_one_epoch, evaluate
#from model import mobile_vit_small as create_model
from model import BiAM,GoogLeNetFeatureExtractor

import warnings
warnings.filterwarnings("ignore")

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
#设置设备为GPU（如果可用）或CPU
    if os.path.exists("./weights") is False:#检查是否存在名为“weights”的目录，如果不存在则创建它。
        os.makedirs("./weights")

    tb_writer = SummaryWriter()#初始化了一个用于写入TensorBoard的SummaryWriter

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)#读取并分割数据为训练集和验证集

    img_size = 224#设置了图像大小为224

    data_transform = {#定义了用于训练和验证集的数据转换，包括将图像转换为灰度图、调整大小、裁剪、翻转、转换为张量以及对像素值进行归一化
        "train": transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                     transforms.Resize(int(img_size)),
                                     transforms.RandomResizedCrop(img_size),
                                     transforms.ToTensor()
                                     ]),
        "val": transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                   transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor()])}

    # 实例化训练数据集
    #通过MyDataSet类实例化了训练数据集train_dataset，并传入了训练数据集的路径、类别以及数据转换方法。
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    #例化了验证数据集val_dataset，同样传入了验证数据集的路径、类别以及数据转换方法。
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])
#设置了批处理大小batch_size，并且将数据集传入torch.utils.data.DataLoader中，创建了train_loader，
    batch_size = args.batch_size
    #nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    #print('Using {} dataloader workers every process'.format(nw))
    nw = 0
    # 设置了批处理大小、是否打乱数据、是否将数据加载到固定内存位置、以及数据加载的并行工作数等参数。
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    #val_loader = torch.utils.data.DataLoader(val_dataset,
    #                                         batch_size=batch_size,
    #                                         shuffle=False,
    #                                         pin_memory=True,
    #                                         num_workers=nw,
    #                                         collate_fn=val_dataset.collate_fn)

    # 创建特征增强模块实例
    model_biam = BiAM().to(device)
    googlenet = models.googlenet(pretrained=True).to(device)
    # 加载预训练的GoogLeNet模型
    # 将模型设置为评估模式
    googlenet.eval()
    # 创建特征提取器实例
    feature_extractor = GoogLeNetFeatureExtractor(googlenet)
    #model = create_model(num_classes=args.num_classes).to(device)
    # 模型的输出类别数量由args.num_classes指定。然后，使用.to(device)将模型移动到之前设置的设备（GPU或CPU）上进行计算。
    #summary(model,input_size=(3, 224, 224))

    #pg = [p for p in model.parameters() if p.requires_grad]
    #定义了优化器和学习率调度器，并在循环中进行训练。
    ## initialize optimizer ###
    optimizer = optim.Adam(model_biam.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=0)
    best_acc = 0
    for epoch in range(args.epochs):
        #在每个epoch中，使用train_one_epoch函数进行模型的训练，并计算训练集的损失和准确率。
        #将训练得到的模型保存到指定的路径中。在训练的过程中，可以使用验证集来评估模型的性能，并记录最佳的准确率。
        # train
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        train_loss, train_acc = train_one_epoch(model=model_biam,
                                                googlenet=feature_extractor,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)


        # validate
        #val_loss, val_acc = evaluate(model=model,
        #                             data_loader=val_loader,
        #                             device=device,
        #                             epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]#创建了一个名为tags的列表，其中包含要记录的指标名称，如训练损失、训练准确率、验证损失、验证准确率和学习率。
        tb_writer.add_scalar(tags[0], train_loss, epoch)#使用tensorboard写入器（tb_writer）将当前epoch的训练损失（train_loss）作为名为"train_loss"的标签的标量值添加到记录中。
        tb_writer.add_scalar(tags[1], train_acc, epoch)#将当前epoch的训练准确率（train_acc）作为名为"train_acc"的标签的标量值添加到记录中
        #tb_writer.add_scalar(tags[2], val_loss, epoch)
        #tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)#将当前epoch的学习率（optimizer.param_groups[0]["lr"]）作为名为"learning_rate"的标签的标量值添加到记录中。
        lr_scheduler.step()#更新学习率调度器以更新优化器的学习率。
        if train_acc > best_acc:#检查当前的训练准确率（train_acc）是否大于迄今为止观察到的最佳准确率（best_acc）。
            # 如果是，则更新最佳准确率，并将模型的状态字典保存到"./weights/best_model.pth"文件中。
            best_acc = train_acc
            torch.save(model_biam.state_dict(), "./weights/best_model1.pth")

        torch.save(model_biam.state_dict(), "./weights/latest_model1.pth")#将模型的状态字典保存到"./weights/latest_model.pth"文件中，无论训练准确率如何。

if __name__ == '__main__':
    # 这段代码使用了Python标准库中的argparse模块，用于解析命令行参数。
    # 解析器对象被创建并存储在变量parser中，接着使用add_argument()方法添加了一些参数。这些参数包括：
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=636)#num_classes：int类型，表示分类数目，默认值为636。
    parser.add_argument('--epochs', type=int, default=400)#epochs：int类型，表示训练轮数，默认值为200。
    parser.add_argument('--batch-size', type=int, default=16)#batch-size：int类型，表示每个批次的大小，默认值为32。
    parser.add_argument('--lr', type=float, default=0.0009)#lr：float类型，表示学习率，默认值为0.0005。

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="./SDUMLA/train/")#data-path：str类型，表示数据集所在根目录，默认值为"./636chuli42/train/"。

    # 预训练权重路径，如果不想载入就设置为空字符
    #parser.add_argument('--weights', type=str, default='./mobilevit_xxs.pt',E:/deeplearning/deep-learning-for-image-processing-master/pytorch_classification/MobileViT/weights/latest_model.pth
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')#weights：str类型，表示预训练权重的路径，默认值为空字符。
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)#freeze-layers：bool类型，表示是否冻结权重，默认值为False。
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')#device：str类型，表示设备ID，默认值为'cuda:0'。


    opt = parser.parse_args()#使用parse_args()方法解析命令行参数，并将结果存储在opt变量中。这样，就可以通过命令行参数来控制程序的行为。

    main(opt)

''''
这段代码是一个用于训练图像分类模型的Python脚本。它使用了PyTorch深度学习框架和一些相关的库和工具。以下是代码的主要功能：

导入所需的库和模块，包括os、argparse、torchsummary、torch等。
定义了一个main函数，该函数接收命令行参数并进行模型训练。
在main函数中，首先设置了设备（GPU或CPU），创建了一个用于记录训练过程的SummaryWriter，并读取了训练和验证数据的路径。
接着定义了数据预处理的转换操作，包括对训练数据进行随机裁剪、水平翻转等操作，对验证数据进行裁剪和归一化等操作。
创建了训练集和验证集的数据加载器，用于将数据传入模型进行训练。
定义了模型结构，并根据命令行参数加载了预训练模型的权重（如果有的话）。
如果指定了冻结部分层的操作，则冻结了除分类器以外的所有层。
设置了优化器和学习率调度器，并开始进行多个epoch的训练，同时记录训练过程中的损失、准确率和学习率等信息。
在每个epoch结束时，保存了当前的模型权重，并在训练准确率提升时保存了最佳模型权重。
整体而言，这段代码是一个完整的图像分类模型训练脚本，包括了数据预处理、模型定义、训练循环和模型保存等功能。
'''
