import os
import time
import torch
import argparse
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from RGBBranch import RGBBranch
from SemBranch import SemBranch
from SASceneNet import SASceneNet
from Libs.Datasets.MITIndoor67Dataset import MITIndoor67Dataset
from Libs.Utils import utils
from Libs.Utils.torchsummary import torchsummary
from Libs.Utils.dfw.dfw import DFW
import numpy as np
import yaml

parser = argparse.ArgumentParser(description='Semantic-Aware Scene Recognition Evaluation')
parser.add_argument('--ConfigPath', metavar='DIR', help='Configuration file path',
                    default='Config/config_MITIndoor.yaml')


def evaluationDataLoader(train_dataloader, valid_dataloader, model, epoch, epoch_valid, model_save_dir,
                         save_model_name):
    # 定义优化器
    # model_optimizer = DFW(model.parameters(), eta=CONFIG['TRAINING']['LR'], momentum=CONFIG['TRAINING']['MOMENTUM'],
    #                       weight_decay=CONFIG['TRAINING']['WEIGHT_DECAY'])
    model_optimizer = torch.optim.SGD(params=model.parameters(), lr=CONFIG['TRAINING']['LR'], momentum=CONFIG['TRAINING']['MOMENTUM'],
                          weight_decay=CONFIG['TRAINING']['WEIGHT_DECAY'])

    for epoch_ in range(epoch):
        batch_time = utils.AverageMeter()
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top2 = utils.AverageMeter()
        top5 = utils.AverageMeter()

        ClassTPs_Top1 = torch.zeros(1, len(classes), dtype=torch.uint8).to(device)
        ClassTPs_Top2 = torch.zeros(1, len(classes), dtype=torch.uint8).to(device)
        ClassTPs_Top5 = torch.zeros(1, len(classes), dtype=torch.uint8).to(device)
        Predictions = np.zeros(len(train_dataloader))
        SceneGTLabels = np.zeros(len(train_dataloader))

        # Extract batch size
        batch_size = CONFIG['TRAINING']['BATCH_SIZE']['TRAIN']

        # Start data time
        data_time_start = time.time()

        ##################################
        # train one epoch
        ##################################
        for i, (mini_batch) in enumerate(train_dataloader):
            mini_batch = next(iter(train_dataloader))
            start_time = time.time()
            # 读取数据
            RGB_image = mini_batch['Image'].to(device)
            semantic_mask = mini_batch['Semantic'].to(device)
            semantic_scores = mini_batch['Semantic Scores'].to(device)
            sceneLabelGT = mini_batch['Scene Index'].to(device)

            # Create tensor of probabilities from semantic_mask
            semanticTensor = utils.make_one_hot(semantic_mask, semantic_scores,
                                                C=CONFIG['DATASET']['N_CLASSES_SEM'])

            # Model Forward
            outputSceneLabel, feature_conv, outputSceneLabelRGB, outputSceneLabelSEM = model(RGB_image,
                                                                                             semanticTensor)
            # Compute class accuracy
            ClassTPs = utils.getclassAccuracy(outputSceneLabel, sceneLabelGT, len(classes), topk=(1, 2, 5))
            ClassTPs_Top1 += ClassTPs[0]
            ClassTPs_Top2 += ClassTPs[1]
            ClassTPs_Top5 += ClassTPs[2]

            # Compute Loss
            loss = model.loss(outputSceneLabel, sceneLabelGT)

            # Model Backward
            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()

            # Measure Top1, Top2 and Top5 accuracy
            prec1, prec2, prec5 = utils.accuracy(outputSceneLabel.data, sceneLabelGT, topk=(1, 2, 5))

            # Update values
            losses.update(loss.item(), batch_size)
            top1.update(prec1.item(), batch_size)
            top2.update(prec2.item(), batch_size)
            top5.update(prec5.item(), batch_size)

            # Measure batch elapsed time
            batch_time.update(time.time() - start_time)

            # Print information
            if i % CONFIG['TRAINING']['PRINT_FREQ'] == 0:
                set = 'Training'
                print('{} set batch: [{}/{}]\t'
                      'Batch Time {batch_time.val:.3f} (avg: {batch_time.avg:.3f})\t'
                      'Loss {loss.val:.3f} (avg: {loss.avg:.3f})\t'
                      'Prec@1 {top1.val:.3f} (avg: {top1.avg:.3f})\t'
                      'Prec@2 {top2.val:.3f} (avg: {top2.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} (avg: {top5.avg:.3f})'.
                      format(set, i, len(train_dataloader), set, batch_time=batch_time, loss=losses,
                             top1=top1, top2=top2, top5=top5))

                print(
                    'Elapsed time for {} set training {time:.3f} seconds'.format(set,
                                                                                 time=time.time() - data_time_start))
                print("")

        ##################################
        # valid
        ##################################
        # 模型验证,每隔step_valid个epoch对所有验证数据进行验证
        set = 'Validation'
        if epoch_ % epoch_valid == 0:
            batch_time = utils.AverageMeter()
            losses = utils.AverageMeter()
            top1 = utils.AverageMeter()
            top2 = utils.AverageMeter()
            top5 = utils.AverageMeter()

            ClassTPs_Top1 = torch.zeros(1, len(classes), dtype=torch.uint8).to(device)
            ClassTPs_Top2 = torch.zeros(1, len(classes), dtype=torch.uint8).to(device)
            ClassTPs_Top5 = torch.zeros(1, len(classes), dtype=torch.uint8).to(device)
            Predictions = np.zeros(len(valid_dataloader))
            SceneGTLabels = np.zeros(len(valid_dataloader))

            # Extract batch size
            batch_size = CONFIG['VALIDATION']['BATCH_SIZE']['TRAIN']

            # Start data time
            data_time_start = time.time()

            with torch.no_grad():
                for i, (mini_batch) in enumerate(valid_dataloader):
                    start_time = time.time()
                    # 读取数据
                    RGB_image = mini_batch['Image'].to(device)
                    semantic_mask = mini_batch['Semantic'].to(device)
                    semantic_scores = mini_batch['Semantic Scores'].to(device)
                    sceneLabelGT = mini_batch['Scene Index'].to(device)

                    if set is 'Validation' and CONFIG['VALIDATION']['TEN_CROPS']:
                        # Fuse batch size and ncrops to set the input for the network
                        bs, ncrops, c_img, h, w = RGB_image.size()
                        RGB_image = RGB_image.view(-1, c_img, h, w)

                        bs, ncrops, c_sem, h, w = semantic_mask.size()
                        semantic_mask = semantic_mask.view(-1, c_sem, h, w)

                        bs, ncrops, c_sem, h, w = semantic_scores.size()
                        semantic_scores = semantic_scores.view(-1, c_sem, h, w)

                    # Create tensor of probabilities from semantic_mask
                    semanticTensor = utils.make_one_hot(semantic_mask, semantic_scores,
                                                        C=CONFIG['DATASET']['N_CLASSES_SEM'])

                    # Model Forward
                    outputSceneLabel, feature_conv, outputSceneLabelRGB, outputSceneLabelSEM = model(RGB_image,
                                                                                                     semanticTensor)

                    if set is 'Validation' and CONFIG['VALIDATION']['TEN_CROPS']:
                        # Average results over the 10 crops
                        outputSceneLabel = outputSceneLabel.view(bs, ncrops, -1).mean(1)
                        outputSceneLabelRGB = outputSceneLabelRGB.view(bs, ncrops, -1).mean(1)
                        outputSceneLabelSEM = outputSceneLabelSEM.view(bs, ncrops, -1).mean(1)

                    if batch_size is 1:
                        if set is 'Validation' and CONFIG['VALIDATION']['TEN_CROPS']:
                            feature_conv = torch.unsqueeze(feature_conv[4, :, :, :], 0)
                            RGB_image = torch.unsqueeze(RGB_image[4, :, :, :], 0)

                        # Obtain 10 most scored predicted scene index
                        Ten_Predictions = utils.obtainPredictedClasses(outputSceneLabel)

                        # Save predicted label and ground-truth label
                        Predictions[i] = Ten_Predictions[0]
                        SceneGTLabels[i] = sceneLabelGT.item()

                        # Compute activation maps
                        # utils.saveActivationMap(model, feature_conv, Ten_Predictions, sceneLabelGT,
                        #                         RGB_image, classes, i, set, save=True)

                    # Compute class accuracy
                    ClassTPs = utils.getclassAccuracy(outputSceneLabel, sceneLabelGT, len(classes), topk=(1, 2, 5))
                    ClassTPs_Top1 += ClassTPs[0]
                    ClassTPs_Top2 += ClassTPs[1]
                    ClassTPs_Top5 += ClassTPs[2]

                    # Compute Loss
                    loss = model.loss(outputSceneLabel, sceneLabelGT)

                    # Measure Top1, Top2 and Top5 accuracy
                    prec1, prec2, prec5 = utils.accuracy(outputSceneLabel.data, sceneLabelGT, topk=(1, 2, 5))

                    # Update values
                    losses.update(loss.item(), batch_size)
                    top1.update(prec1.item(), batch_size)
                    top2.update(prec2.item(), batch_size)
                    top5.update(prec5.item(), batch_size)

                    # Measure batch elapsed time
                    batch_time.update(time.time() - start_time)

                    # Print information
                    if i % CONFIG['VALIDATION']['PRINT_FREQ'] == 0:
                        print('{} set batch: [{}/{}]\t'
                              'Batch Time {batch_time.val:.3f} (avg: {batch_time.avg:.3f})\t'
                              'Loss {loss.val:.3f} (avg: {loss.avg:.3f})\t'
                              'Prec@1 {top1.val:.3f} (avg: {top1.avg:.3f})\t'
                              'Prec@2 {top2.val:.3f} (avg: {top2.avg:.3f})\t'
                              'Prec@5 {top5.val:.3f} (avg: {top5.avg:.3f})'.
                              format(set, i, len(valid_dataloader), set, batch_time=batch_time, loss=losses,
                                     top1=top1, top2=top2, top5=top5))

                        print(
                            'Elapsed time for {} set training {time:.3f} seconds'.format(set,
                                                                                         time=time.time() - data_time_start))
                        print("")
        # 保存每轮模型
        state = {"state_dict": model.state_dict(), "epoch": epoch_, "train_loss": losses.sum,
                 'best_prec1': top1.val}
        torch.save(state, os.path.join(model_save_dir, "epoch_%d_" % epoch_ + save_model_name))


global device, classes, CONFIG

# Decode CONFIG file information
args = parser.parse_args()
CONFIG = yaml.safe_load(open(args.ConfigPath, 'r'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 是否使用GPU训练

print('-' * 65)
print("Training starting...")
print('-' * 65)

# Instantiate network
if CONFIG['MODEL']['ONLY_RGB']:
    print('Training ONLY RGB branch')
    print('Selected RGB backbone architecture: ' + CONFIG['MODEL']['ARCH'])
    model = RGBBranch(arch=CONFIG['MODEL']['ARCH'], scene_classes=CONFIG['DATASET']['N_CLASSES_SCENE'])
elif CONFIG['MODEL']['ONLY_SEM']:
    print('Training ONLY SEM branch')
    model = SemBranch(scene_classes=CONFIG['DATASET']['N_CLASSES_SCENE'],
                      semantic_classes=CONFIG['DATASET']['N_CLASSES_SEM'])
else:
    print('Training complete model')
    print('Selected RG backbone architecture: ' + CONFIG['MODEL']['ARCH'])
    model = SASceneNet(arch=CONFIG['MODEL']['ARCH'], scene_classes=CONFIG['DATASET']['N_CLASSES_SCENE'],
                       semantic_classes=CONFIG['DATASET']['N_CLASSES_SEM'])

# Move Model to GPU an set it to evaluation mode
model = model.to(device)

print('-' * 65)
print('Loading dataset {}...'.format(CONFIG['DATASET']['NAME']))

traindir = os.path.join(CONFIG['DATASET']['ROOT'], CONFIG['DATASET']['NAME'])
valdir = os.path.join(CONFIG['DATASET']['ROOT'], CONFIG['DATASET']['NAME'])

if CONFIG['DATASET']['NAME'] == "MITIndoor67":
    train_dataset = MITIndoor67Dataset(traindir, "train")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG['TRAINING']['BATCH_SIZE']['TRAIN'],
                                               shuffle=True, num_workers=CONFIG['DATALOADER']['NUM_WORKERS'],
                                               pin_memory=True)

    val_dataset = MITIndoor67Dataset(valdir, "val", tencrops=CONFIG['VALIDATION']['TEN_CROPS'], SemRGB=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=CONFIG['VALIDATION']['BATCH_SIZE']['TRAIN'],
                                             shuffle=False,
                                             num_workers=CONFIG['DATALOADER']['NUM_WORKERS'], pin_memory=True)

    classes = train_dataset.classes

# Print dataset information
print('Dataset loaded!')
print('Dataset Information:')
print('Train set. Size {}. Batch size {}. Nbatches {}'
      .format(len(train_loader) * CONFIG['TRAINING']['BATCH_SIZE']['TRAIN'],
              CONFIG['TRAINING']['BATCH_SIZE']['TRAIN'], len(train_loader)))
print('Validation set. Size {}. Batch size {}. Nbatches {}'
      .format(len(val_loader) * CONFIG['VALIDATION']['BATCH_SIZE']['TRAIN'],
              CONFIG['VALIDATION']['BATCH_SIZE']['TRAIN'],
              len(val_loader)))
print('Train set number of scenes: {}'.format(len(classes)))
print('Validation set number of scenes: {}'.format(len(classes)))

print('-' * 65)

print('Computing histogram of scene classes...')

TrainHist = utils.getHistogramOfClasses(train_loader, classes, "Training")
ValHist = utils.getHistogramOfClasses(val_loader, classes, "Validation")

# Print Network information
print('-' * 65)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('Number of params: {}'.format(params))
print('-' * 65)
print('GPU in use: {} with {} memory'.format(torch.cuda.get_device_name(0), torch.cuda.max_memory_allocated(0)))
print('-' * 65)

# Summary of the network for a dummy input
torchsummary.summary(model, [(3, 224, 224), (CONFIG['DATASET']['N_CLASSES_SEM'] + 1, 224, 224)],
                     batch_size=CONFIG['TRAINING']['BATCH_SIZE']['TRAIN'])

##############################################################################
print('Training on dataset ...')

# Train model on training and validation set
EPOCH = 30  # 迭代轮数
epoch_valid = 1  # 验证的频率：每隔epoch_valid进行验证
model_save_dir = 'Data/Model Zoo/MIT_Self_Train/'
save_model_name = CONFIG['MODEL']['NAME'] + '.pth'
evaluationDataLoader(train_dataloader=train_loader, valid_dataloader=val_loader, model=model, epoch=EPOCH,
                     epoch_valid=epoch_valid, model_save_dir=model_save_dir, save_model_name=save_model_name)

##############################################################################
