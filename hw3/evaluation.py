import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable

import config
import data
import utils
from btm_up_top_dwn import Net


def evaluate_hw3():
    # set mannual seed
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    val_loader = data.get_loader(val=True)

    cudnn.benchmark = True

    question_keys = val_loader.dataset.vocab['question'].keys()
    net = Net(question_keys)
    net.load_state_dict(torch.load('model.pkl', map_location=lambda storage, loc: storage))
    net = nn.DataParallel(net).cuda()  # Support multiple GPUS
    net.eval()

    accs = []
    for v, q, a, b, idx, v_mask, q_mask, q_len in val_loader:
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
        acc = utils.batch_accuracy(out, answer).data.cpu()
        accs.append(acc.mean().item())

    return sum(accs) / len(accs)

