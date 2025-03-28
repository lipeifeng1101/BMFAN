import sys
import json
import os
import torch
import numpy
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision import transforms, datasets
import torchvision.models as models
#from model import mobile_vit_xx_small as create_model
from model7 import BiAM,GoogLeNetFeatureExtractor
import warnings
warnings.filterwarnings("ignore")


#33 + 0.9967948717948718
#33   0.992521

#33 3 0.9871 gai 0.9690

#42 0.9967
#42 0.9983
#636 42 + 0.9952830188679245     -0.9929
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #检查是否有可用的 CUDA 设备，如果有则使用 CUDA，否则使用 CPU。
    img_size = 224 #设置图像的大小为 224x224 像素。
    data_transform = { #定义了数据预处理的操作，包括将图像转换为灰度图、调整大小、中心裁剪、转换为张量并进行归一化。
                 "val": transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                   transforms.Resize(int(img_size)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor()])}

    #image_path = 'E:/deeplearning/deep-learning-for-image-processing-master/data/'
    image_path = './SDUMLA/' #设置图像数据集的路径。


    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"]) #创建了用于验证的图像数据集，并应用了预处理操作。

    batch_size = 6 #设置了批处理的大小
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers 确定了用于数据加载的工作线程数。
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    # read class_indict
    json_path = './class_indices.json' #置了类别索引的 JSON 文件路径。
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = BiAM().to(device) #创建了一个名为 Resnext 的模型实例，并将其移动到指定的设备上。
    googlenet = models.googlenet(pretrained=True).to(device)
    # 加载预训练的GoogLeNet模型
    # 将模型设置为评估模式
    googlenet.eval()
    # 创建特征提取器实例
    feature_extractor = GoogLeNetFeatureExtractor(googlenet)
    # load model weights
    weights_path = "./weights/best_model1.pth" #设置了模型权重文件的路径。
    assert os.path.exists(weights_path), "file: '{}' does not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path)) # 加载了预训练的模型权重。

    model.eval() # 设置模型为评估模式

    acc = 0.0  # accumulate accurate number / epoch初始化一个变量用于累积每个 epoch 的准确数量。
    pre_out = []  # 初始化两个空列表，用于存储模型输出和标签。
    pre_out_lable = []
    with torch.no_grad():  # 使用 torch.no_grad() 上下文管理器，确保在验证过程中不会进行梯度计算，以节省内存和提高速度。
        val_bar = tqdm(validate_loader, file=sys.stdout)  # 使用 tqdm 创建一个进度条，用于迭代验证数据集，并将进度信息输出到标准输出。
        preds = []  # 初始化两个空列表，用于存储模型预测结果。
        preds_01 = []
        for val_data in val_bar:  # 遍历验证数据集中的每个数据。
            val_images, val_labels = val_data  # 解包验证数据中的图像和标签。
            outputs = model(val_images.to(device),feature_extractor )  # eval model only have last output layer将验证图像传递给模型进行推理，得到模型的输出结果。
            # 将模型输出和标签分别添加到 pre_out 和 pre_out_lable 列表中。
            pre_out.append(outputs)
            pre_out_lable.append(val_labels)

            predict = torch.softmax(outputs, dim=0)  # 对模型输出进行 softmax 操作，得到预测概率。
            predict_cla = torch.argmax(predict, dim=1)  # 根据预测概率取得最大概率对应的类别。
            predict_print = predict.cpu().numpy()  # 将预测概率和模型输出转换为 NumPy 数组。
            predict_print_1 = outputs.cpu().numpy()

            predict_print_01 = numpy.sum(predict_print)  # 计算了预测概率的和以及归一化后的预测概率，并将结果存储到 preds_01 列表中。
            predict_print_temp = predict_print / predict_print_01
            preds_01.append(predict_print_temp)

            preds.append(predict_print_1)  # 将模型输出结果添加到 preds 列表中。
            # print(predict_print)
            predict_y = torch.max(outputs, dim=1)[1]  # 根据模型输出取得最大值对应的类别
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()  # 计算并累积每个 batch 的预测准确数量。

        # print(preds.__sizeof__())
        predictions = numpy.concatenate(preds, axis=0)  # 将所有预测结果和归一化后的预测概率连接成一个大数组。
        predictions_01 = numpy.concatenate(preds_01, axis=0)

        numpy.save('./temp_3', predictions)  # 将预测结果和归一化后的预测概率保存到文件中。
        numpy.save('./temp01_3', predictions_01)

    loss_E = torch.cat(pre_out, dim=0)  # 将模型输出和标签连接成张量。
    loss_E_lable = torch.cat(pre_out_lable, dim=0)
    torch.save(loss_E, 'tensor_2.pt')  # 将模型输出和标签保存到文件中。
    torch.save(loss_E_lable, 'tensor_liable_2.pt')
    # loss_EER_tem = EER_loss.eer_loss(loss_E,loss_E_lable).to(device)
    # 计算并打印了验证集的准确率
    val_accurate = acc / val_num
    print(val_accurate)
    # print("sum_loss_eer:" + str(sum_loss_eer))
    # print("sum_loss_eer:" + str(loss_EER_tem))


if __name__ == '__main__':
    main()
