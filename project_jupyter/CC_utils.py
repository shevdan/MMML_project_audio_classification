
from scipy.sparse import csr_matrix
import numpy as np
from scipy.signal import resample
from scipy.fftpack import dct

EarQ = 9.26449
minBW = 24.7

class CQ_Spectrogram:
    def __init__(self, sample_rate=22050, n_fft=512, hop_length=0.01, win_length=0.025,
        window='hamming', low_freq=0, high_freq=None, number_of_octaves=7, number_of_bins_per_octave=24,
        spectral_threshold = 0.005, fundamental_freq = 120, q_rate = 1.0,
        center=True, pad_mode='reflect', power=2.0, freeze_parameters=True):
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.sample_rate = sample_rate
        self.low_freq = low_freq
        if high_freq is None:
            high_freq = sample_rate // 2
        self.high_freq = high_freq
        
        self.number_of_octaves = number_of_octaves
        self.number_of_bins_per_octave = number_of_bins_per_octave
        self.spectral_threshold = spectral_threshold
        self.fundamental_freq = fundamental_freq
        self.q_rate = q_rate

    
    def framing(self, input: np.ndarray):
        frame_length = int(self.win_length * self.sample_rate)
        frame_step = int(self.hop_length * self.sample_rate)


        # stride trick: https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html
        a = np.array(input)
        nrows = ((a.size - frame_length) // frame_step) + 1
        n = a.strides[0]
        frames =  np.lib.stride_tricks.as_strided(
            a, shape=(nrows, frame_length), strides=(frame_step * n, n)
        )

        if len(frames[-1]) < frame_length:
            frames[-1] = np.append(
                frames[-1], np.array([0] * (frame_length - len(frames[0])))
            )

        return frames, frame_length
        

    def windowing(self, frames, frame_len):
        return {
            "hanning": np.hanning(frame_len) * frames,
            "bartlet": np.bartlett(frame_len) * frames,
            "kaiser": np.kaiser(frame_len, beta=14) * frames,
            "blackman": np.blackman(frame_len) * frames,
            "hamming": np.hamming(frame_len) * frames,
        }[self.window]
    
    def spectrogram(self, windows):
        tmp_cqt_freqs = np.array(
            [
                self.fundamental_freq * 2 ** ((m * self.number_of_bins_per_octave + n) / self.number_of_bins_per_octave)
                for m in range(self.number_of_octaves)
                for n in range(self.number_of_bins_per_octave)
            ]
        )
        cqt_freqs = tmp_cqt_freqs[
            (self.low_freq <= tmp_cqt_freqs) & (tmp_cqt_freqs <= self.high_freq)
        ]

        # calculate Q
        Q = self.q_rate / (2 ** (1.0 / self.number_of_bins_per_octave) - 1.0)

        # compute Nks (win_lens)
        win_lens = np.ceil(Q * self.sample_rate / cqt_freqs).astype(np.int64)
        win_lens = win_lens[win_lens <= self.n_fft]

        # filter center freqs and count number of pitches & frames
        cqt_freqs = cqt_freqs[-1 * len(win_lens) :]
        n_pitch = len(cqt_freqs)
        n_frames = len(windows)

        # calculate kernel
        a = np.zeros((n_pitch, self.n_fft), dtype=np.complex128)
        kernel = np.zeros(a.shape, dtype=np.complex128)

        for k in range(n_pitch):
            Nk = win_lens[k]
            fk = cqt_freqs[k]

            # prepare indices
            start_index = int((self.n_fft - Nk) / 2)
            end_index = start_index + Nk

            # prepare kernel
            temp_a = np.exp(2.0 * np.pi * 1j * (fk / self.sample_rate) * np.arange(0, Nk))
            a[k, start_index:end_index] = (1 / Nk) * self.windowing(temp_a, Nk)
            kernel[k] = np.fft.fft(a[k], self.n_fft)

        # prepare sparse computation vars
        kernel[np.abs(kernel) <= self.spectral_threshold] = 0.0
        kernel_sparse = csr_matrix(kernel).conjugate() / self.n_fft

        # compute transform
        spec = np.zeros([n_frames, n_pitch], dtype=np.complex128)
        for k, frame in enumerate(windows):
            x = (
                np.r_[frame, np.zeros(self.n_fft - len(frame))]
                if len(frame) < self.n_fft
                else frame[0 : len(frame)]
            )
            spec[k] = np.fft.fft(x, self.n_fft) * kernel_sparse.T
        return spec.T
    

    def cqt_transform(self, input):
        frames, frame_length = self.framing(input)
        windows = self.windowing(frames, frame_length)
        cqt_tr = self.spectrogram(windows)
        return cqt_tr


class CQCC:
    def __init__(self, sample_rate=22050, n_fft=512, hop_length=0.01, win_length=0.025,
        window='hamming', low_freq=0, high_freq=None, number_of_octaves=7, number_of_bins_per_octave=24,
        spectral_threshold = 0.005, fundamental_freq = 120, q_rate = 1.0, num_ceps=13,
        resampling_ratio = 0.95,  n_mels=64, fmin=0.0, fmax=None, 
        is_log=True, ref=1.0, amin=1e-10, top_db=80.0, freeze_parameters=True):
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.sample_rate = sample_rate
        self.low_freq = low_freq
        if high_freq is None:
            high_freq = sample_rate // 2
        
        self.number_of_octaves = number_of_octaves
        self.number_of_bins_per_octave = number_of_bins_per_octave
        self.spectral_threshold = spectral_threshold
        self.fundamental_freq = fundamental_freq
        self.q_rate = q_rate

        self.num_ceps = num_ceps
        self.resampling_ratio = resampling_ratio

    
    def cqcc(self, input):
        # |Xcq|**2
        power_spectrum = np.absolute(input) ** 2

        # -> log(.)
        # handle zeros: if feat is zero, we get problems with log
        features_no_zero = np.where(power_spectrum == 0, np.finfo(float).eps, power_spectrum)
        log_features = np.log(features_no_zero)

        # uniform resampling
        resampled_features = resample(
            log_features, int(len(log_features) * self.resampling_ratio)
        )

        #  -> DCT(.)
        cqccs = dct(x=resampled_features, type=2, axis=1, norm="ortho")[:, :self.num_ceps]


        return cqccs
    


class GC_Spectrogram:
    def __init__(self, sample_rate=22050, n_fft=512, hop_length=0.01, win_length=0.025,
        window='hamming', low_freq=0, high_freq=None, nfilts=24,
        center=True, pad_mode='reflect', power=2.0, freeze_parameters=True):

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.sample_rate = sample_rate
        self.low_freq = low_freq
        if high_freq is None:
            high_freq = sample_rate // 2
        self.high_freq = high_freq
        
        self.nfilts = nfilts

    def framing(self, input: np.ndarray):
        frame_length = int(self.win_length * self.sample_rate)
        frame_step = int(self.hop_length * self.sample_rate)


        # stride trick: https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html
        a = np.array(input)
        nrows = ((a.size - frame_length) // frame_step) + 1
        n = a.strides[0]
        frames =  np.lib.stride_tricks.as_strided(
            a, shape=(nrows, frame_length), strides=(frame_step * n, n)
        )

        if len(frames[-1]) < frame_length:
            frames[-1] = np.append(
                frames[-1], np.array([0] * (frame_length - len(frames[0])))
            )

        return frames, frame_length
        

    def windowing(self, frames, frame_len):
        return {
            "hanning": np.hanning(frame_len) * frames,
            "bartlet": np.bartlett(frame_len) * frames,
            "kaiser": np.kaiser(frame_len, beta=14) * frames,
            "blackman": np.blackman(frame_len) * frames,
            "hamming": np.hamming(frame_len) * frames,
        }[self.window]


    def generate_center_frequencies(self, min_freq: float, max_freq: float, nfilts: int) -> np.ndarray:
        """
        Compute center frequencies in the ERB scale.

        Args:
            min_freq (float) : minimum frequency of the center frequencies' domain.
            max_freq (float) : maximum frequency of the center frequencies' domain.
            nfilts     (int) : number of filters <=> number of center frequencies to compute.

        Returns:
            (numpy.ndarray) : array of center frequencies.
        """
        # init vars
        m = np.array(range(nfilts)) + 1
        c = EarQ * minBW

        # compute center frequencies
        center_freqs = (max_freq + c) * np.exp(
            (m / nfilts) * (np.log(min_freq + c) - np.log(max_freq + c))
        ) - c
        return center_freqs[::-1]
    

    def compute_gain(
        self, fcs: np.ndarray, B: np.ndarray, wT: np.ndarray, T: float
    ) :
        """
        Compute Gain and matrixify computation for speed purposes [Ellis-spectrogram]_.

        Args:
            fcs (numpy.ndarray) : center frequencies in
            B   (numpy.ndarray) : bandwidths of the filters.
            wT  (numpy.ndarray) : corresponds to :code:`(omega)*T = 2*pi*freq*T`.
            T           (float) : periode in seconds aka inverse of the sampling rate.

        Returns:
            (tuple):
                - (numpy.ndarray) : a 2d numpy array representing the filter gains.
                - (numpy.ndarray) : a 2d array A used for final computations.
        """
        # pre-computations for simplification
        K = np.exp(B * T)
        Cos = np.cos(2 * fcs * np.pi * T)
        Sin = np.sin(2 * fcs * np.pi * T)
        Smax = np.sqrt(3 + 2 ** (3 / 2))
        Smin = np.sqrt(3 - 2 ** (3 / 2))

        # define A matrix rows
        A11 = (Cos + Smax * Sin) / K
        A12 = (Cos - Smax * Sin) / K
        A13 = (Cos + Smin * Sin) / K
        A14 = (Cos - Smin * Sin) / K

        # Compute gain (vectorized)
        A = np.array([A11, A12, A13, A14])
        Kj = np.exp(1j * wT)
        Kjmat = np.array([Kj, Kj, Kj, Kj]).T
        G = 2 * T * Kjmat * (A.T - Kjmat)
        Coe = -2 / K**2 - 2 * Kj**2 + 2 * (1 + Kj**2) / K
        Gain = np.abs(G[:, 0] * G[:, 1] * G[:, 2] * G[:, 3] * Coe**-4)
        return A, Gain

    def hz2erb(self, f):
        A = (1000 * np.log(10)) / (24.7 * 4.37)
        return A * np.log10(1 + f * 0.00437)

    def gammatone_filter_banks(self, fs):

        order = 4

        # define custom difference func
        def Dif(u, a):
            return u - a.reshape(self.nfilts, 1)

        # init vars
        fbank = np.zeros([self.nfilts, self.n_fft])
        width = 1.0
        maxlen = self.n_fft // 2 + 1
        T = 1 / fs
        n = 4
        u = np.exp(1j * 2 * np.pi * np.array(range(self.n_fft // 2 + 1)) / self.n_fft)
        idx = range(self.n_fft // 2 + 1)

        # computer center frequencies, convert to ERB scale and compute bandwidths
        fcs = self.generate_center_frequencies(self.low_freq, self.high_freq, self.nfilts)
        ERB = width * ((fcs / EarQ) ** order + minBW**order) ** (1 / order)
        B = 1.019 * 2 * np.pi * ERB

        # compute input vars
        wT = 2 * fcs * np.pi * T
        pole = np.exp(1j * wT) / np.exp(B * T)

        # compute gain and A matrix
        A, Gain = self.compute_gain(fcs, B, wT, T)

        # compute fbank
        fbank[:, idx] = (
            (T**4 / Gain.reshape(self.nfilts, 1))
            * np.abs(Dif(u, A[0]) * Dif(u, A[1]) * Dif(u, A[2]) * Dif(u, A[3]))
            * np.abs(Dif(u, pole) * Dif(u, pole.conj())) ** (-n)
        )

        # make sure all filters has max value = 1.0
        try:
            fbank = np.array([f / np.max(f) for f in fbank[:, range(maxlen)]])

        except BaseException:
            fbank = fbank[:, idx]

        # compute scaling
        scaling = np.ones(shape=(self.nfilts, 1))
        fbank = fbank * scaling
        return fbank, np.array([self.hz2erb(freq) for freq in fcs])

        
    def spectrogram(self, input, fs):
        fbanks, _ = self.gammatone_filter_banks(fs)

        frames, frame_length = self.framing(input)
        windows = self.windowing(frames, frame_length)

        fourrier_transform = np.absolute(np.fft.fft(windows, self.n_fft))
        fourrier_transform = fourrier_transform[:, : int(self.n_fft / 2) + 1]

        ## Power Spectrum
        abs_fft_values = (1.0 / self.n_fft) * np.square(fourrier_transform)

        #  -> x Gammatone-fbanks
        features = np.dot(abs_fft_values, fbanks.T)
        return features, fourrier_transform


class GFCC:
    def __init__(self, sample_rate=22050, n_fft=512, hop_length=0.01, win_length=0.025,
        window='hamming', low_freq=0, high_freq=None, number_of_octaves=7, number_of_bins_per_octave=24,
        spectral_threshold = 0.005, fundamental_freq = 120, q_rate = 1.0, num_ceps=13,
        resampling_ratio = 0.95,  n_mels=64, fmin=0.0, fmax=None, 
        is_log=True, ref=1.0, amin=1e-10, top_db=80.0, freeze_parameters=True):
    
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.sample_rate = sample_rate
        self.low_freq = low_freq
        if high_freq is None:
            high_freq = sample_rate // 2
        
        self.number_of_octaves = number_of_octaves
        self.number_of_bins_per_octave = number_of_bins_per_octave
        self.spectral_threshold = spectral_threshold
        self.fundamental_freq = fundamental_freq
        self.q_rate = q_rate

        self.num_ceps = num_ceps
        self.resampling_ratio = resampling_ratio


    def gfcc(self, input):
        nonlin_rect_features = np.power(input, 1 / 3)
        gfccs = dct(x=nonlin_rect_features, type=2, axis=1, norm="ortho")[
            :, :self.num_ceps
        ]


        return gfccs


if __name__== "__main__":
    import scipy.io.wavfile
    import spafe.utils.vis as vis
    from spafe.features.cqcc import cqcc
    from spafe.features.gfcc import gfcc, erb_spectrogram
    fs, sig = scipy.io.wavfile.read('samples/blues.00009.wav')
    print(sig.shape)
    print(fs)
    sp = CQ_Spectrogram()
    spectogram = sp.cqt_transform(sig)
    cq = CQCC()
    cqcc1 = cq.cqcc(spectogram)

    cqccs = cqcc(sig=sig, fs=fs, pre_emph=False)


    gf_sp = GC_Spectrogram()
    spectogram2, ft = gf_sp.spectrogram(sig)
    gf = GFCC()
    gfcc1 = gf.gfcc(spectogram2)

    feats, fourrier_transform = erb_spectrogram(sig=sig, fs=fs, pre_emph=False)


    # print(spectogram2[0] == feats[0])


    gfccs = gfcc(sig=sig, fs=fs, pre_emph=False)

    # print(gfcc1[0])
    # print(gfccs[0])


    # print(gfcc1 == gfccs)

    # print(cqcc1 == cqccs)
    # print(cqcc1[0])
    # print(cqccs[0])

    # vis.visualize(cqcc, 'LMFCC Coefficient Index','Frame Index')
