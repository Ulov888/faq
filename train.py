# encoding: utf-8
import warnings

warnings.filterwarnings('ignore')

import transformers
transformers.logging.set_verbosity_error()

import os
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from transformers import get_linear_schedule_with_warmup

from models import TextMatchingSIModel,FGM
from prepare_datasets import tokenizer, DataLoaderX, CustomDatasetPointwise, \
    collate_to_max_length_pointwise_si
from eval_util import load_raw_data, qa_acc

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./log/tec/1')



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(101)


parser = argparse.ArgumentParser(description='SI text matching model training.')
parser.add_argument('--pretrained_model_path', default='./pretrained_model/bert_pytorch',
                    type=str, help='model save path')
parser.add_argument('--pooling_strategy', default='cls', type=str,
                    help='pooling strategy type')
parser.add_argument('--batch_size', default=32, type=int, help='train batch size')
parser.add_argument('--num_epochs', default=20, type=int, help='train epochs')
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--train_file', default='./tec_platform_data/mixed_data/train.txt', type=str, help='train file')
parser.add_argument('--dev_file', default='./tec_platform_data/mixed_data/dev.txt', type=str, help='dev file')
parser.add_argument('--test_file1', default='./tec_platform_data/tec_test.json', type=str, help='test file')
parser.add_argument('--test_file2', default='', type=str, help='test file')
parser.add_argument('--output_dir', default='./save_models_tec/', type=str, help='model save dir')
args = parser.parse_args()


custom_dataset_train = CustomDatasetPointwise(args.train_file, is_training=True)
train_loader = DataLoaderX(dataset=custom_dataset_train,
                           batch_size=args.batch_size,
                           shuffle=True,
                           collate_fn=collate_to_max_length_pointwise_si,
                           num_workers=8,
                           pin_memory=True)

custom_dataset_dev = CustomDatasetPointwise(args.dev_file, is_training=False)
dev_loader = DataLoaderX(dataset=custom_dataset_dev,
                         batch_size=args.batch_size,
                         shuffle=False,
                         collate_fn=collate_to_max_length_pointwise_si,
                         num_workers=8,
                         pin_memory=True)

net = TextMatchingSIModel(args.pretrained_model_path,
                          pooling_strategy=args.pooling_strategy)



# ckpt = torch.load('save_models_tec/it_model_bert_cls_9879.pth')
# net.load_state_dict(ckpt)

best_acc = 0.
best_qa_success_acc = 0.

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

ce_loss = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=args.lr)
len_dataset = len(custom_dataset_train)
total_steps = (len_dataset // args.batch_size) * args.num_epochs if len_dataset % args.batch_size == 0 \
    else (len_dataset // args.batch_size + 1) * args.num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*total_steps,
                                            num_training_steps=total_steps)


def train(epoch):
    print('\nTrain Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    fgm = FGM(net)
    for batch_index, batch_input in enumerate(train_loader):
        for i in range(len(batch_input)):
            batch_input[i] = batch_input[i].to(device)
        inputs = batch_input[0:3]
        targets = batch_input[-1]

        outputs = net(*inputs)
        loss = ce_loss(outputs, targets)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)

        fgm.attack()
        loss_adv = ce_loss(outputs, targets)
        loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        fgm.restore()  # 恢复embedding参数

        optimizer.step()
        net.zero_grad()

        scheduler.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_index % 50 == 0:
            print("Current train loss %.4f | train acc %.4f" % (train_loss / (batch_index + 1), correct / total))
        writer.add_scalar('loss_batch_index', train_loss / (batch_index + 1), batch_index)

    writer.add_scalar('train_loss', train_loss/(batch_index+1), epoch)
    print("Current epoch train loss %.4f | train acc %.4f" % (train_loss / (batch_index + 1), correct / total))
    writer.add_scalar('accuracy/train_acc', correct / total, epoch)


def test(epoch):
    global best_acc, best_qa_success_acc
    print('\nTest Epoch: %d' % epoch)
    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_index, batch_input in enumerate(dev_loader):
            for i in range(len(batch_input)):
                batch_input[i] = batch_input[i].to(device)
            inputs = batch_input[0:3]
            targets = batch_input[-1]

            outputs = net(*inputs)
            loss = ce_loss(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        cur_acc = round(correct / total, 4)
        print("Current epoch val loss %.4f | val acc %.4f" % (val_loss / (batch_index + 1), cur_acc))
    writer.add_scalar('accuracy/val_acc', cur_acc, epoch)
    writer.add_scalar('val_loss', val_loss/(batch_index+1), epoch)


    # if cur_acc <= best_acc:
    #     return
    test_acc = 0.
    for test_file in [args.test_file1]:
        print("++++++++++++++++++++++++++++++++++++++++++")
        querys, all_recalls, test_data_y, test_data_qids = load_raw_data(test_file)
        with torch.no_grad():
            logits = []
            num_batch = int(len(querys) / args.batch_size) + 1
            for i in range(num_batch):
                batch_texts1 = querys[args.batch_size * i:args.batch_size * (i + 1)]
                batch_texts2 = all_recalls[args.batch_size * i:args.batch_size * (i + 1)]
                if len(batch_texts1) > 0:
                    source = tokenizer(batch_texts1, batch_texts2, add_special_tokens=True,
                                       truncation=True,
                                       padding=True, return_tensors='pt',
                                       max_length=64,
                                       return_attention_mask=True,
                                       return_token_type_ids=True)

                    input_ids = source.get('input_ids').to(device)
                    attention_mask = source.get('attention_mask').to(device)
                    token_type_ids = source.get('token_type_ids').to(device)
                    result = net(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                    logits.append(result.cpu().numpy())
            logits = np.concatenate(logits, axis=0)
            logits_max = np.max(logits, axis=-1, keepdims=True)
            logits_exp = np.exp(logits - logits_max)
            logits_sum = np.sum(logits_exp, axis=-1, keepdims=True)
            probs = logits_exp / logits_sum
            probs = probs[:, -1]
            sim_thds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            for sim_thd in sim_thds:
                recall_rate, qa_success_rate, qa_error_rate, qa_no_reply_rate, \
                recall_top1, precision_top1, f1_top1, pos_sizes, pos_correct_sizes, \
                neg_sizes, neg_correct_sizes, pos_acc, neg_acc, overall_acc = qa_acc(test_data_y, probs,
                                                                                     test_data_qids, sim_thd)

                if sim_thd==0.9 and qa_success_rate > best_acc:
                    test_acc = qa_success_rate
    writer.add_scalar('accuracy/test_acc', qa_success_rate, epoch)

    # Save checkpoint.
    if cur_acc > best_acc:
        print('Saving model..')
        if not os.path.isdir(args.output_dir):
            os.mkdir(args.output_dir)
        net_to_save = net.module if hasattr(net, 'module') else net
        model_type = args.pretrained_model_path.split('/')[-1].split('_')[0]
        torch.save(net_to_save.state_dict(), os.path.join(args.output_dir,
                                                          'mixed_model_{}_{}.pth'.format(model_type,
                                                                                   args.pooling_strategy)))
        best_acc = cur_acc
        best_qa_success_acc = qa_success_rate


def main():
    for epoch in range(0, args.num_epochs):
        train(epoch)
        test(epoch)


if __name__ == '__main__':
    pass
    main()

