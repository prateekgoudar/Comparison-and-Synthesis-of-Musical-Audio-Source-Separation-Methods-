#!/usr/bin/env python
# coding: utf-8

# In[43]:


import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
audio, sr = librosa.load("bass.wav", sr=16000)
spectrogram = librosa.stft(audio, n_fft = 2048, hop_length = 256, center = False, win_length = 2048)
Mm = abs (spectrogram)
phase = spectrogram / (Mm + 2.2204e-16)
librosa.display.specshow (Mm)
print(Mm.shape)
Mm = Mm[:, 0:3904]
print(Mm.shape)
phase = phase[:, 0:3904]
print(phase.shape)


# In[44]:


audio2, sr2 = librosa.load("vocals.wav", sr=16000)
spectrogram2 = librosa.stft(audio2, n_fft = 2048, hop_length = 256, center = False, win_length = 2048)
Ms = abs (spectrogram2)
phase2 = spectrogram2 / (Ms + 2.2204e-16)
librosa.display.specshow (Ms)
print(Ms.shape)
Ms = Ms[:, 0:3904]
print(Mm.shape)
phase2 = phase2[:, 0:3904]
print(phase2.shape)


# In[45]:


audio3, sr3 = librosa.load("drums.wav", sr=16000)
spectrogram3 = librosa.stft(audio3, n_fft = 2048, hop_length = 256, center = False, win_length = 2048)
Md = abs (spectrogram3)
phase3 = spectrogram3 / (Md + 2.2204e-16)
librosa.display.specshow (Md)
print(Md.shape)
Md = Md[:, 0:3904]
print(Md.shape)
phase3 = phase3[:, 0:3904]
print(phase3.shape)


# In[47]:


audio4, sr4 = librosa.load("other.wav", sr=16000)
spectrogram4 = librosa.stft(audio4, n_fft = 2048, hop_length = 256, center = False, win_length = 2048)
Mo = abs (spectrogram4)
phase4 = spectrogram4 / (Mo + 2.2204e-16)
librosa.display.specshow (Mo)
print(Mo.shape)
Mo = Mo[:, 0:3904]
print(Mo.shape)
phase4 = phase4[:, 0:3904]
print(phase4.shape)


# In[48]:


print(Mm.shape)
print(Ms.shape)
print(Md.shape)
print(Mo.shape)


# In[49]:


def NMF_Train(M, B_init, W_init, n_iter):
    B = B_init
    W = W_init
    KLD_1 = np.zeros((6))
    for i in range(0, n_iter+1):
        KLD = 0
        M_hat = np.matmul(B,W)
        for j in range (0, M.shape[0]):
            for k in range (0, M.shape[1]):
                KLD+= (M[j,k] * np.log(M[j,k] / M_hat[j,k])) - M[j,k] + M_hat[j,k]
        B = np.multiply(B, np.divide((np.matmul(np.divide(M, M_hat), W.T)),(np.matmul(np.ones((M.shape[0],M.shape[1])), W.T))))
        W = np.multiply(W, np.divide((np.matmul(B.T, np.divide(M, M_hat))),(np.matmul(B.T, (np.ones((M.shape[0],M.shape[1])))))))
        if i == 0:
            KLD_1[0] = KLD
            print (f'The value of KL Divergence for number of iterations {i} is {KLD}')
        elif i == 50:
            KLD_1[1] = KLD
            print (f'The value of KL Divergence for number of iterations {i} is {KLD}')
        elif i == 100:
            KLD_1[2] = KLD
            print (f'The value of KL Divergence for number of iterations {i} is {KLD}')
        elif i == 150:
            KLD_1[3] = KLD
            print (f'The value of KL Divergence for number of iterations {i} is {KLD}')
        elif i == 200:
            KLD_1[4] = KLD
            print (f'The value of KL Divergence for number of iterations {i} is {KLD}')
        elif i == 250:
            KLD_1[5] = KLD
            print (f'The value of KL Divergence for number of iterations {i} is {KLD}')
        else:
            continue
        
    if n_iter == 100:
        #print (f'The value of KL Divergence for number of iterations 250 is {KLD_1[5]}')
        n = np.array([0,50,100,150,200,250])
        plt.figure()
        plt.title(f'KL Divergence vs number of iterations') 
        plt.xlabel('Number of iterations')
        plt.ylabel('KL Divergence')        
        plt.plot(n,KLD_1)
        plt.show()
    return B, W


# In[50]:


import csv
with open('Bm_init.csv', 'r') as file1:
    f1 = list(csv.reader(file1, delimiter=','))
    Bm_init = np.array(f1, dtype=np.float)
with open('Bs_init.csv', 'r') as file2:
    f2 = list(csv.reader(file2, delimiter=','))
    Bs_init = np.array(f2, dtype=np.float)
with open('Wm_init.csv', 'r') as file3:
    f3 = list(csv.reader(file3, delimiter=','))
    Wm_init = np.array(f3, dtype=np.float)
    Wm_init = np.delete(Wm_init, 976, axis=1)
with open('Ws_init.csv', 'r') as file4:
    f4 = list(csv.reader(file4, delimiter=','))
    Ws_init = np.array(f4, dtype=np.float)
    Ws_init = np.delete(Ws_init, 976, axis=1)


# In[51]:


print(Bm_init.shape, Wm_init.shape)


# In[52]:


Wm_init2 = np.append(Wm_init, Wm_init, axis=1)
Wm_init3 = np.append(Wm_init2, Wm_init2, axis=1)
print(Bm_init.shape, Wm_init3.shape)
Ws_init2 = np.append(Ws_init, Ws_init, axis=1)
Ws_init3 = np.append(Ws_init2, Ws_init2, axis=1)
print(Bs_init.shape, Ws_init3.shape)


# In[53]:


M_hat2 = np.matmul(Bm_init,Wm_init3)
print (M_hat2.shape)


# In[11]:


b_1, w_1 = NMF_Train(Mm, Bm_init, Wm_init,10)
print (b_1)
print (w_1)


# In[38]:


b_1, w_1 = NMF_Train(Mm, Bm_init, Wm_init3,100)


# In[39]:


b_2, w_2 = NMF_Train(Ms, Bs_init, Ws_init3,100)


# In[54]:


b_3, w_3 = NMF_Train(Md, Bm_init, Wm_init3,100)


# In[55]:


b_4, w_4 = NMF_Train(Mo, Bm_init, Wm_init3,100)


# In[56]:


print(b_1.shape)
print(b_2.shape)
print(b_3.shape)
print(b_4.shape)


# In[57]:


def separate_signals(M, B_vocals, B_drums,B_bass, B_other, n_iter):
    B_mixed = np.concatenate((B_vocals, B_drums, B_bass, B_other), axis=1)
    #W_mixed = np.concatenate((w_2, w_1), axis=0)
    W_mixed = np.random.rand(B_mixed.shape[1],M.shape[1])
    for i in range(0, n_iter):
        M_mixed_hat = np.matmul(B_mixed,W_mixed)
        W_mixed = np.multiply(W_mixed, np.divide((np.matmul(B_mixed.T, np.divide(M, M_mixed_hat))),(np.matmul(B_mixed.T, (np.ones((M.shape[0],M.shape[1])))))))
    W_mixed1 = np.zeros((B_vocals.shape[1], M.shape[1]))
    W_mixed2 = np.zeros((B_drums.shape[1], M.shape[1]))
    W_mixed3 = np.zeros((B_bass.shape[1], M.shape[1]))
    W_mixed4 = np.zeros((B_other.shape[1], M.shape[1]))
    for j in range (0, B_vocals.shape[1]):
        W_mixed1[j,:] = W_mixed[j,:]
    for k in range (0, B_drums.shape[1]):
        W_mixed2[k,:] = W_mixed[j,:]
        j+= 1
    for y in range (0, B_bass.shape[1]):
        W_mixed3[y,:] = W_mixed[j,:]
        j+= 1
    for z in range (0, B_other.shape[1]):
        W_mixed4[z,:] = W_mixed[j,:]
        j+= 1
        
    M_vocals = np.matmul(B_vocals,W_mixed1)
    M_drums = np.matmul(B_drums,W_mixed2)
    M_bass = np.matmul(B_bass,W_mixed3)
    M_other = np.matmul(B_other,W_mixed4)
    return M_vocals, M_drums, M_bass, M_other


# In[59]:


audio3, sr3 = librosa.load("mixture.wav", sr=16000)
spectrogram3 = librosa.stft(audio3, n_fft = 2048, hop_length = 256, center = False, win_length = 2048)
M_mixed = abs (spectrogram3)
phase3 = spectrogram3 / (M_mixed + 2.2204e-16)
print(M_mixed.shape)
M_mixed_short = M_mixed[:, 0:3904]
print(M_mixed_short.shape)
M_vocals_rec, M_drums_rec, M_bass_rec, M_other_rec = separate_signals(M_mixed_short, b_2, b_3, b_1, b_4, 500)

with open ('M_vocals_rec_AnimalTiger.csv', 'w')as f_2:
    writeintocsv_2 = csv.writer (f_2, delimiter = ",")
    writeintocsv_2.writerows(M_vocals_rec)
    
with open ('M_drums_rec_AnimalTiger.csv', 'w')as f_3:
    writeintocsv_3 = csv.writer (f_3, delimiter = ",")
    writeintocsv_3.writerows(M_drums_rec)
    
with open ('M_bass_rec_AnimalTiger.csv', 'w')as f_4:
    writeintocsv_4 = csv.writer (f_4, delimiter = ",")
    writeintocsv_4.writerows(M_bass_rec)
    
with open ('M_other_rec_AnimalTiger.csv', 'w')as f_5:
    writeintocsv_5 = csv.writer (f_5, delimiter = ",")
    writeintocsv_5.writerows(M_other_rec)

signal_1_hat = librosa.istft(M_vocals_rec * phase2, hop_length=256, win_length=2048) 
librosa.output.write_wav("vocals_rec.wav", signal_1_hat, sr=16000)
signal_2_hat = librosa.istft(M_drums_rec * phase3[:, 0:3904], hop_length=256, win_length=2048) 
librosa.output.write_wav("drums_rec.wav", signal_2_hat, sr=16000)
signal_3_hat = librosa.istft(M_bass_rec * phase, hop_length=256, win_length=2048) 
librosa.output.write_wav("bass_rec.wav", signal_3_hat, sr=16000)
signal_4_hat = librosa.istft(M_other_rec * phase4, hop_length=256, win_length=2048) 
librosa.output.write_wav("other_rec.wav", signal_4_hat, sr=16000)


# In[ ]:




