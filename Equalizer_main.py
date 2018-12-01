import sys
import pyaudio
import numpy as np
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication,QDialog
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QMessageBox , QSlider , QLabel
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import os
import matplotlib.pyplot as plt
import pandas as pd
from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets
import matplotlib as mpl
from scipy import signal
from scipy.io.wavfile import read
from scipy.io.wavfile import write
import wave
from scipy.signal import firwin,freqz
from scipy.fftpack import fft, ifft
import pyqtgraph as pg
import struct
import matplotlib
#matplotlib.use('TkAgg')
from tkinter import TclError
import queue
#get_ipython().run_line_magic('matplotlib', 'tk')
dir = os.getcwd()
sys.path.insert(0, dir)
fileName = ''
db = [0] * 8
sampling_rate = 0.0
class Equalizer(QDialog):
    def __init__(self):
        super().__init__()
        loadUi('equalizer.ui',self)
        self.setWindowTitle('Task 4 DSP')
        self.Browse_Button.clicked.connect(self.browsefunc)
        self.Play_button.clicked.connect(self.play_sound)
        self.S_1.setMaximum(30)
        self.S_1.setMinimum(-30)
        self.S_1.setSliderPosition(0)
        self.S_1.sliderReleased.connect(self.S_1_)
        self.S_2.setMaximum(30)
        self.S_2.setMinimum(-30)
        self.S_2.setSliderPosition(0)
        self.S_2.sliderReleased.connect(self.S_2_)
        self.S_3.setMaximum(30)
        self.S_3.setMinimum(-30)
        self.S_3.setSliderPosition(0)
        self.S_3.sliderReleased.connect(self.S_3_)
        self.S_4.setMaximum(30)
        self.S_4.setMinimum(-30)
        self.S_4.setSliderPosition(0)
        self.S_4.sliderReleased.connect(self.S_4_)
        self.S_5.setMaximum(30)
        self.S_5.setMinimum(-30)
        self.S_5.setSliderPosition(0)
        self.S_5.sliderReleased.connect(self.S_5_)
        self.S_6.setMaximum(30)
        self.S_6.setMinimum(-30)
        self.S_6.setSliderPosition(0)
        self.S_6.sliderReleased.connect(self.S_6_)
        self.S_7.setMaximum(30)
        self.S_7.setMinimum(-30)
        self.S_7.setSliderPosition(0)
        self.S_7.sliderReleased.connect(self.S_7_)
        self.S_8.setMaximum(30)
        self.S_8.setMinimum(-30)
        self.S_8.setSliderPosition(0)
        self.S_8.sliderReleased.connect(self.S_8_)
        self.live_sound.clicked.connect(self.live)
    @pyqtSlot()
    def browsefunc(self):
        try:
            global hamm, s_, step
            global dir
            global fileName
            global db
            global signal
            fileName_ = QFileDialog.getOpenFileName(self, 'Select Signal', dir,
                                                    '*.wav')
            fileName = fileName_[0]
            if fileName_[1] != '*.wav':
                QMessageBox.about(self, "Error!", "Choose a wav file")
                return
            spf = wave.open(fileName, 'rb')
            global samp_width
            samp_width = spf.getsampwidth()
            frames = spf.getnframes()
            signal = spf.readframes(frames)
            signal = np.fromstring(signal, 'Int16')
            global sampling_rate
            global filtered_signal
            sampling_rate = spf.getframerate()
            print(fileName)

            s = fft(signal)
            step = int(len(s)/16)
            print('s shape is')
            print(np.shape(s))
            lowcut= [0,step,(2*step),(3*step),(4*step),(5*step),(6*step),(7*step),(8*step),(9*step),(10*step),(11*step),(12*step),(13*step),(14*step),(15*step),(16*step)]
            s_ = np.empty((step,16),dtype= complex)
            for i in range(0,16):
                s_[:,i] = s[lowcut[i]:lowcut[i+1]]
            hamm = np.hamming(step)

            global filtered
            filtered = np.empty((step, 16), dtype=complex)
            for x in range(0, 16):
                if x <= 7 :
                    filtered[:,x] = np.multiply(hamm,s_[:,x])  * 10**(db[x] / 10.0)
                else :
                    filtered[:,x] = np.multiply(hamm,s_[:,x])  * 10**(db[int(15-x)] / 10.0)


            filtered_signal = np.concatenate([filtered[0:step,0],filtered[0:step,1],filtered[0:step,2],filtered[0:step,3],filtered[0:step,4],filtered[0:step,5],filtered[0:step,6],filtered[0:step,7],filtered[0:step,8],filtered[0:step,9],filtered[0:step,10],filtered[0:step,11],filtered[0:step,12],filtered[0:step,13],filtered[0:step,14],filtered[0:step,15]])
            print(np.shape(filtered_signal))
            global filtered_signal_t
            filtered_signal_t = ifft(filtered_signal)

            filtered_signal_t = np.round(filtered_signal_t).astype('int16')
            print(filtered_signal_t)

            self.Filtered_view.clear

            self.Filtered_view.plot(filtered_signal_t)
            self.Filtered_freq.clear
            self.Filtered_freq.plot(abs(filtered_signal.real))
            self.Signal_freq.clear
            self.Signal_freq.plot(abs(s.real))
            self.Signal_view.clear
            self.Signal_view.plot(signal)
        except:
            QMessageBox.about(self, "Error!", "Choose a wav file")
            return

    def S_1_(self):
        global db
        s = self.S_1.value()
        db[0] = float(s)
        self.label_1.setText(str(db[0]) + ' dB')
        self.Filter(0)


    def S_2_(self):
        global db
        s = self.S_2.value()
        db[1] = float(s)
        self.label_2.setText(str(db[1]) + ' dB')
        self.Filter(1)


    def S_3_(self):
        global db
        s = self.S_3.value()
        db[2] = float(s)
        self.label_3.setText(str(db[2]) + ' dB')
        self.Filter(2)


    def S_4_(self):
        global db
        s = self.S_4.value()
        db[3] = float(s)
        self.label_4.setText(str(db[3]) + ' dB')
        self.Filter(3)


    def S_5_(self):
        global db
        s = self.S_5.value()
        db[4] = float(s)
        self.label_5.setText(str(db[4]) + ' dB')
        self.Filter(4)


    def S_6_(self):
        global db
        s= self.S_6.value()
        db[5] = float(s)
        self.label_6.setText(str(db[5]) + ' dB')
        self.Filter(5)


    def S_7_(self):
        global db
        s = self.S_7.value()
        db[6] = float(s)
        self.label_7.setText(str(db[6]) + ' dB')
        self.Filter(6)


    def S_8_(self):
        global db
        s = self.S_8.value()
        db[7] = float(s)
        self.label_8.setText(str(db[7]) + ' dB')
        self.Filter(7)


    def Filter(self,x):
        try:
            global filtered_signal_t
            global sampling_rate
            global db
            global signal
            global hamm , s_, step
            hamm = np.hamming(step)

            print(db)
            global filtered

            global filtered_signal
            filtered[:, x] = np.multiply(hamm, s_[:, x]) * 10 ** (db[x] / 10.0)
            filtered[:, 15-x] = np.multiply(hamm, s_[:, 15-x]) * 10 ** (db[x] / 10.0)

            filtered_signal = np.concatenate(
                [filtered[0:step, 0], filtered[0:step, 1], filtered[0:step, 2], filtered[0:step, 3], filtered[0:step, 4],
                 filtered[0:step, 5], filtered[0:step, 6], filtered[0:step, 7], filtered[0:step, 8], filtered[0:step, 9],
                 filtered[0:step, 10], filtered[0:step, 11], filtered[0:step, 12], filtered[0:step, 13],
                 filtered[0:step, 14], filtered[0:step, 15]])
            print(np.shape(filtered_signal))

            filtered_signal_t = ifft(filtered_signal)

            filtered_signal_t = np.round(filtered_signal_t).astype('int16')
            print(filtered_signal_t)

            self.Filtered_view.clear()
            self.Filtered_view.plot(filtered_signal_t)

            self.Filtered_freq.clear()
            self.Filtered_freq.plot(abs(filtered_signal.real))
            print("DONE!")
        except:
            QMessageBox.about(self, "Error!", "Choose a wav file first")
            return

    def play_sound(self):
        try:
            global samp_width
            global filtered_signal_t
            p = pyaudio.PyAudio()
            stream = p.open(format=p.get_format_from_width(samp_width),
                            channels=1,
                            rate=int(2*sampling_rate),
                            output=True)
            stream.write(filtered_signal_t)
            stream.stop_stream()
            stream.close()

            p.terminate()
        except:
            QMessageBox.about(self, "Error!", "Choose a wav file first")
            return

    def live(self):
        import matplotlib.pyplot as plt
        CHUNK = 4069  # samples per frame
        FORMAT = pyaudio.paInt16  # audio format (bytes per sample?)
        CHANNELS = 2  # single channel for microphone
        RATE = 44100  # samples per second

        # create matplotlib figure and axes
        fig = plt.figure(figsize=(15, 10))
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        fig.show()
        fig.canvas.draw()

        # pyaudio class instance
        p = pyaudio.PyAudio()

        # stream object to get data from microphone
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            output=False,
            frames_per_buffer=CHUNK
        )

        # variable for plotting
        x = np.arange(0, 2 * CHUNK, 2)  # samples (waveform)
        y = np.arange(0, 2 * CHUNK, 2)  # samples (waveform)

        # create a line object with random data
        line, = ax1.plot(x, np.random.rand(CHUNK), '-', lw=2)

        # create a line object with random data
        line2, = ax2.plot(y, np.random.rand(CHUNK), '-', lw=2)

        # format waveform axes
        ax1.set_title('AUDIO WAVEFORM')
        ax1.set_xlabel('samples')
        ax1.set_ylabel('volume')
        ax1.set_ylim(0, 255)
        ax1.set_xlim(0, 2 * CHUNK)
        plt.setp(ax1, xticks=[0, CHUNK, 2 * CHUNK], yticks=[0, 128, 255])

        # format waveform axes
        ax2.set_title('AUDIO filter WAVEFORM')
        ax2.set_xlabel('samples')
        ax2.set_ylabel('volume')
        ax2.set_ylim(0, 255)
        ax2.set_xlim(0, 2 * CHUNK)
        plt.setp(ax2, xticks=[0, CHUNK, 2 * CHUNK], yticks=[0, 128, 255])

        x0 = 0
        x1 = 0
        x2 = 0
        y0 = 0
        y1 = 0
        y2 = 0
        print('stream start')

        while True:
            out = []
            # binary data
            data = stream.read(CHUNK)
            # print(len(data))
            # convert data to integers, make np array, then offset it by 127
            data_int = struct.unpack(str(4 * CHUNK) + 'B', data)
            # data_int = np.fromstring(data, dtype=np.int16)
            # create np array and offset by 128
            data_np = np.array(data_int, dtype='b')[::4] + 128

            line.set_ydata(data_np)
            temp0 = data_np
            for i in range(len(temp0)):
                x0 = x1
                x1 = x2
                x2 = temp0[i]
                y0 = y1
                y1 = y2
                # y2 = 2* r * cos((Low_Cut off  frequency)*y1) - (r^2 * y0) + (gain*(x2  - x0))

                # y[n] = b0 x[n] + b1 x[n-1] + b2 x[n-2] - a1 y[n-1] - a2 y[n-2]
                y2 = (3 / 8) * x2 + (-3 / 8) * x0 - (-0.3) * y1 - (0.25) * y0 + 128
                out.append(y2)
                if i == 4068:
                    line2.set_ydata(out)

            # update figure canvas
            try:
                fig.canvas.draw()
                fig.canvas.flush_events()
            except TclError:
                print('stream stopped')
                break

        # In[5]:
'''''''''
        from scipy import signal

        b = [3 / 8, 0, -3 / 8]
        a = [1, -0.3, 0.25]
        w, h = signal.freqz(b, a)

        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.title('Digital filter frequency response')
        ax1 = fig.add_subplot(111)

        plt.plot(w, 20 * np.log10(abs(h)), 'b')
        plt.ylabel('Amplitude [dB]', color='b')
        plt.xlabel('Frequency [rad/sample]')

        ax2 = ax1.twinx()
        angles = np.unwrap(np.angle(h))
        plt.plot(w, angles, 'g')
        plt.ylabel('Angle (radians)', color='g')
        plt.grid()
        plt.axis('tight')
        plt.show()
'''''''''''

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = Equalizer()
    widget.show()
    sys.exit(app.exec_())