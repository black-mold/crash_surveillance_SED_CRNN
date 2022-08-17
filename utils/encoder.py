import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import librosa
import torch
import scipy


class multi_label_encoder:
    """
    role : label encode
    """
    def __init__(self, fs = 16000, audio_len = 60, n_fft = 640, hop_length = 80, net_pooling = 1):
        self.fs = fs # 나중에 바꿔줘야 함, 사용 안함.
        self.audio_len = audio_len # unit : second

        self.hop_length = hop_length   #mel-spectrogram
        self.n_fft = n_fft             #mel-spectrogram
        
        self.net_pooling = net_pooling  # input frame -> output frame 에서 어떤 비율로 감소하는지. 
        # 1이면 input frame = outputframe
        # 2면   input frame = 2 * output frame

        n_samples = self.audio_len * self.fs

        self.n_frames = int(int((n_samples/ self.hop_length)) / self.net_pooling)

    def _time_to_frame(self, time):
        samples = time * self.fs
        frame = (samples) / self.hop_length
        return np.clip(frame / self.net_pooling, a_min=0, a_max=self.n_frames)


    def encoder_strong_label(self, xml_dir):
        """Encode a list of strong label"""
        xml = ET.parse(xml_dir)
        root = xml.getroot()

        item = root.find("events").findall("item")


        onset = [float(x.findtext("STARTSECOND")) for x in item] # list, str
        offset = [float(x.findtext("ENDSECOND")) for x in item]  # list, str
        label_idx = [int(x.findtext("CLASS_ID")) for x in item] 

        target = np.zeros([self.n_frames, 3], dtype = 'float32')  # shape : [frame, class], class : 3



        if (len(onset) != len(offset)): 
            print("wrong")
        else:
            for i in range(len(onset)):
                start = int(self._time_to_frame(onset[i])) #버림 -> 해당 time frame에 걸쳐있으면 true??
                end = int(np.ceil(self._time_to_frame(offset[i])))   #올림 -> 해당 time frame에 걸쳐있으면 true
                target[start:end, (label_idx[i]-1)] = 1 # (class_id[i]-1) = 1 : scream, 2 : tire skidding, 3 : car crash
            

        return (target)


def matplotlib_label_show(target):
    n_frame = target.shape[0] # target.shape[0] = segment

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    x = np.arange(n_frame)  #(60,)
    y1 = target[:,0]        #(60,) 
    y2 = target[:,1]        #(60,) tire skidding
    y3 = target[:,2]        #(60,) car crash

    ax1.plot(x, y1, label = 'scream')
    ax1.plot(x, y2, label = 'tire skidding', color = 'orange')
    ax1.plot(x, y3, label = 'car crash', color = 'green')


    ax1.legend(framealpha = 1, loc = 'lower right', bbox_to_anchor=(1.4,0))
    # ax1.set_title('{}'.format(xmlfile))


    plt.show()


def result_show(spec, target, pred, post, i, title=None, ylabel='freq_bin', aspect='auto', xmax=None):

    spec = spec.cpu().squeeze()
    target = target.cpu().squeeze()
    pred = pred.detach().cpu().squeeze()
    post = post.detach().cpu().squeeze()

    n_frame = target.shape[0] # target.shape[0] = segment

    fig = plt.figure()

    #spectrogram
    ax1 = plt.subplot(221)
    ax1.set_title(title or 'Spectrogram (db)')
    ax1.set_ylabel(ylabel)
    ax1.set_xlabel('frame')
    im = ax1.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
    if xmax:
        ax1.set_xlim((0, xmax))
    fig.colorbar(im, ax=ax1)



    # label
    ax2 = fig.add_subplot(222)

    x = np.arange(n_frame)  #(60,)
    y1 = target[:,0]        #(60,) 
    y2 = target[:,1]        #(60,) tire skidding
    y3 = target[:,2]        #(60,) car crash

    ax2.plot(x, y1, label = 'scream')
    ax2.plot(x, y2, label = 'tire skidding', color = 'orange')
    ax2.plot(x, y3, label = 'car crash', color = 'green')
    ax2.set_title('label')

    ax3 = fig.add_subplot(223)

    y1 = pred[:,0]        #(60,) 
    y2 = pred[:,1]        #(60,) tire skidding
    y3 = pred[:,2]        #(60,) car crash

    ax3.plot(x, y1, label = 'scream')
    ax3.plot(x, y2, label = 'tire skidding', color = 'orange')
    ax3.plot(x, y3, label = 'car crash', color = 'green')
    ax3.set_title('prediction')

    ax4 = fig.add_subplot(224)

    y1 = post[:,0]        #(60,) 
    y2 = post[:,1]        #(60,) tire skidding
    y3 = post[:,2]        #(60,) car crash

    ax4.plot(x, y1, label = 'scream')
    ax4.plot(x, y2, label = 'tire skidding', color = 'orange')
    ax4.plot(x, y3, label = 'car crash', color = 'green')
    ax4.set_title('post')

    ax3.legend(framealpha = 1, loc = 'lower right', bbox_to_anchor=(2,0))

    fig.subplots_adjust(hspace=0.5)
    fig.suptitle(f'{i+1}th audio')


    plt.show()


def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1)
  axs.set_title(title or 'Spectrogram (db)')
  axs.set_ylabel(ylabel)
  axs.set_xlabel('frame')
  im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  fig.colorbar(im, ax=axs)
  plt.show(block=False)

# def fig2np(fig):
#     data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
#     data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#     return data


def median_filter(pred, median_window, thresholds):
    # pred : [batch, frame, class]
    pred = pred.squeeze() # [frame, class]
    post_filter = scipy.ndimage.filters.median_filter(pred.cpu().detach().numpy(), (median_window, 1)) # [frames, class]
    post_filter = (post_filter>thresholds).astype(float)
    
    return torch.Tensor(post_filter).unsqueeze(0) #  [batch, frame, class] &tensor, float, cpu