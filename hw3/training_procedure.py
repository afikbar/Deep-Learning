import matplotlib
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

import config
import data
import utils
from btm_up_top_dwn import Net

matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sn



def run(net, loader, optimizer, scheduler, tracker, train=False, prefix='', epoch=0):
    """ Run an epoch over the given loader """
    if train:
        net.train()
        # tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    else:
        net.eval()

    tracker_class, tracker_params = tracker.MeanMonitor, {}

    # set learning rate decay policy
    if epoch < len(config.gradual_warmup_steps) and config.schedule_method == 'warm_up':
        utils.set_lr(optimizer, config.gradual_warmup_steps[epoch])

    elif (epoch in config.lr_decay_epochs) and train and config.schedule_method == 'warm_up':
        utils.decay_lr(optimizer, config.lr_decay_rate)

    utils.print_lr(optimizer, prefix, epoch)

    loader = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
    loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))

    for v, q, a, b, idx, v_mask, q_mask, q_len in loader:
        var_params = {
            'requires_grad': False,
        }
        v = Variable(v.cuda(), **var_params)
        q = Variable(q.cuda(), **var_params)
        a = Variable(a.cuda(), **var_params)
        b = Variable(b.cuda(), **var_params)
        q_len = Variable(q_len.cuda(), **var_params)
        v_mask = Variable(v_mask.cuda(), **var_params)
        q_mask = Variable(q_mask.cuda(), **var_params)

        out = net(v, b, q, v_mask, q_mask, q_len)

        answer = utils.process_answer(a)
        loss = utils.calculate_loss(answer, out, method=config.loss_method)
        acc = utils.batch_accuracy(out, answer).data.cpu()

        if train:
            optimizer.zero_grad()
            loss.backward()
            # clip gradient
            clip_grad_norm_(net.parameters(), config.clip_value)
            optimizer.step()
            if config.schedule_method == 'batch_decay':
                scheduler.step()

        loss_tracker.append(loss.item())
        acc_tracker.append(acc.mean())
        fmt = '{:.4f}'.format
        loader.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))

    return acc_tracker.mean.value, loss_tracker.mean.value


def main():
    print('-' * 50)
    config.print_param()

    # set mannual seed
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    train_loader = data.get_loader(train=True)
    val_loader = data.get_loader(val=True)

    cudnn.benchmark = True

    question_keys = val_loader.dataset.vocab['question'].keys()
    net = Net(question_keys)
    net = nn.DataParallel(net).cuda()  # Support multiple GPUS
    select_optim = optim.Adamax if (config.optim_method == 'Adamax') else optim.Adam
    optimizer = select_optim([p for p in net.parameters() if p.requires_grad], lr=config.initial_lr,
                             weight_decay=config.weight_decay)
    scheduler = lr_scheduler.ExponentialLR(optimizer, 0.5 ** (1 / config.lr_halflife))
    print(net)

    train_errors_list = []
    train_losses_list = []

    val_errors_list = []
    val_losses_list = []
    tracker = utils.Tracker()
    for i in range(1, config.epochs + 1):
        # Train:
        train_acc, train_loss = run(net, train_loader, optimizer, scheduler, tracker, train=True, prefix='train',
                                    epoch=i)
        # Val:
        val_acc, val_loss = run(net, val_loader, optimizer, scheduler, tracker, train=False, prefix='val', epoch=i)

        train_errors_list.append(1 - train_acc)
        train_losses_list.append(train_loss)

        val_errors_list.append(1 - val_acc)
        val_losses_list.append(val_loss)

    torch.save(net.module.state_dict(), 'model.pkl')

    sn.set()
    ind = list(range(1, config.epochs + 1))
    # Error
    plt.plot(ind, train_errors_list, label='Train')
    plt.plot(ind, val_errors_list, label='Validation')
    plt.title('Error-rate during epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Error-rate')
    plt.legend()
    # plt.show()
    plt.savefig('error_rate.png')

    plt.clf()

    # Loss
    plt.plot(ind, train_losses_list, label='Train')
    plt.plot(ind, val_losses_list, label='Validation')
    plt.title('Loss during epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()
    plt.savefig('loss_rate.png')


if __name__ == '__main__':
    main()
