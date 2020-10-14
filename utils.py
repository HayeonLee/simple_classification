# import matplotlib
# matplotlib.use('Agg')
import time
import torch
import os


def str2bool(s):
  return s.lower() in ['true', 't']


def str2intlist(s):
  s = s.split(',')
  return [int(v) for v in s]


def str2strlist(s):
  s = s.split(',')
  return [str(v) for v in s]


def cuda(x):
  device = torch.device("cuda:0")
  if torch.cuda.is_available():
    x = x.to(device)
  return x


class Accumulator():
  def __init__(self, *args):
    self.args = args
    self.argdict = {}
    for i, arg in enumerate(args):
      self.argdict[arg] = i
    self.sums = [0] * len(args)
    self.cnt = 0

  def accum(self, val):
    val = [val] if type(val) is not list else val
    val = [v for v in val if v is not None]
    assert (len(val) == len(self.args))
    for i in range(len(val)):
      if torch.is_tensor(val[i]):
        val[i] = val[i].item()
      self.sums[i] += val[i]
    self.cnt += 1

  def clear(self):
    self.sums = [0] * len(self.args)
    self.cnt = 0

  def get(self, arg, avg=True):
    i = self.argdict.get(arg, -1)
    assert (i is not -1)
    return (self.sums[i] / (self.cnt + 1e-8) if avg else self.sums[i])

  def print_(self, header=None, time=None,
             logfile=None, do_not_print=[], as_int=[],
             avg=True):
    msg = '' if header is None else header + ': '
    if time is not None:
      msg += ('(%.3f secs), ' % time)

    args = [arg for arg in self.args if arg not in do_not_print]
    arg = []
    for arg in args:
      val = self.sums[self.argdict[arg]]
      if avg:
        val /= (self.cnt + 1e-8)
      if arg in as_int:
        msg += ('%s %d, ' % (arg, int(val)))
      else:
        msg += ('%s %.4f, ' % (arg, val))
    print(msg)

    if logfile is not None:
      logfile.write(msg + '\n')
      logfile.flush()

  def add_scalars(self, summary, header=None,
                  tag_scalar=None, step=None, avg=True, args=None):
    for arg in self.args:
      val = self.sums[self.argdict[arg]]
      if avg:
        val /= (self.cnt + 1e-8)
      else:
        val = val
      tag = f'{header}/{arg}' if header is not None else arg
      if tag_scalar is not None:
        summary.add_scalars(main_tag=tag,
                            tag_scalar_dict={tag_scalar: val},
                            global_step=step)
      else:
        summary.add_scalar(tag=tag,
                           scalar_value=val,
                           global_step=step)


class Log:
  def __init__(self, args, logf, summary=None):
    self.args = args
    self.logf = logf
    self.summary = summary
    self.stime = time.time()


  def print(self, logger, epoch, tag=None, avg=True):
    ct = time.time() - self.stime
    msg = f'({ct:.4f}s) epoch {epoch} '
    print(msg)
    self.logf.write(msg)
    logger.print_(header=tag, logfile=self.logf, avg=avg)

    if self.summary is not None:
      logger.add_scalars(self.summary, header=tag, step=epoch, avg=avg)
    logger.clear()

    # if tag == 'val':
    #   acc = logger.get('acc', avg=True)
    #   with open(os.path.join(self.exp_name,
    #         f'{self.mode}_rwd_{self.episode}.log'), 'a+') as f:
    #     f.write(f'{acc}\n')
    #     f.flush()

  def print_args(self):
    argdict = vars(self.args)
    print(argdict)
    for k, v in argdict.items():
      self.logf.write(k + ': ' + str(v) + '\n')
    self.logf.write('\n')

  def set_time(self):
    self.stime = time.time()

