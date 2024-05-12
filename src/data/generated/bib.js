define({ entries : {
    "Akbari2015": {
        "abstract": "One important problem in musical information retrieval is automatic music transcription, which is an automated conversion process from played music to a symbolic notation such as MIDI file. Since the accuracy of previous audio-based transcription systems is not satisfactory, we propose an innovative computer vision-based automatic music transcription system named claVision to perform piano music transcription. Instead of processing the music audio, the system performs the transcription only from the video performance captured by a camera mounted over the piano keyboard. In this paper, we describe the architecture and the algorithms used in claVision. The claVision system has a high accuracy (F1 score over 0.95) and a very low latency (about 7.0 ms) in real-time music transcription, even under different illumination conditions. This technology can also be used for other musical keyboard instruments.",
        "author": "Mohammad Akbari and Howard Cheng",
        "doi": "10.1109/TMM.2015.2473702",
        "keywords": "Automatic music transcription, claVision, Computer vision, Multipitch estimation, Piano",
        "title": "Real-Time Piano Music Transcription Based on Computer Vision",
        "type": "ARTICLE",
        "url": "https://ieeexplore.ieee.org/document/7225173",
        "year": "2015"
    },
    "Akbari2018": {
        "abstract": "In order to deal with the challenges arising from acoustic-based music information retrieval such as automatic music transcription, the video of the musical performances can be utilized. In this paper, a new real-time learning-based system for visually transcribing piano music using the CNN-SVM classification of the pressed black and white keys is presented. The whole process in this technique is based on visual analysis of the piano keyboard and the pianist\u2019s hands and fingers. A high accuracy with an average\u00a0F1\u00a0score of 0.95 even under non-ideal camera view, hand coverage, and lighting conditions is achieved. The proposed system has a low latency (about 20 ms) in real-time music transcription. In addition, a new dataset for visual transcription of piano music is created and made available to researchers in this area. Since not all possible varying patterns of the data used in our work are available, an online learning approach is applied to efficiently update the original model based on the new data added to the training dataset.",
        "author": "Mohammad Akbari, Jie Liang and Howard Cheng",
        "doi": "10.1007/s11042-018-5803-1",
        "keywords": "Music information retrieval, Real-time piano music transcription, Image and video processing, Convolutional neural networks, Support vector machines, Online learning",
        "title": "A real-time system for online learning-based visual transcription of piano music",
        "type": "ARTICLE",
        "url": "https://link.springer.com/article/10.1007/s11042-018-5803-1",
        "year": "2018"
    },
    "Cheuk2021": {
        "abstract": "Most of the current supervised automatic music transcription (AMT) models lack the ability to generalize. This means that they have trouble transcribing real-world music recordings from diverse musical genres that are not presented in the labelled training data. In this paper, we propose a semi-supervised framework, ReconVAT, which solves this issue by leveraging the huge amount of available unlabelled music recordings. The proposed ReconVAT uses reconstruction loss and virtual adversarial training. When combined with existing U-net models for AMT, ReconVAT achieves competitive results on common benchmark datasets such as MAPS and MusicNet. For example, in the few-shot setting for the string part version of MusicNet, ReconVAT achieves F1-scores of 61.0% and 41.6% for the note-wise and note-with-offset-wise metrics respectively, which translates into an improvement of 22.2% and 62.5% compared to the supervised baseline model. Our proposed framework also demonstrates the potential of continual learning on new data, which could be useful in real-world applications whereby new data is constantly available.",
        "author": "Kin Wai Cheuk, Dorien Herremans and Li Su",
        "doi": "10.1145/3474085.3475405",
        "keywords": "semi-supervised training, virtual adversarial training, audio processing, automatic music transcription, music information retrieval",
        "title": "ReconVAT: A Semi-Supervised Automatic Music Transcription Framework for Low-Resource Real-World Data",
        "type": "ARTICLE",
        "url": "https://dl.acm.org/doi/10.1145/3474085.3475405",
        "year": "2021"
    },
    "Galin2017": {
        "abstract": "To date, the accurate Music Transcription has been considered to be a rather abstract concept. Our goal was to explore this field of study, which until today has been not attended to. In order to be able to transcribe music automatically, a new type of algorithm has been created. The present dissertation represents the continuation on the Authors research field introduced in Galin A. et al (2015). This algorithm is composed of the Beat Tracking, Note Acquisition and Post Processing processes; during which development, multiple simulations have been run in order to perfect them. With these simultaneously running algorithms, we have reached our goal and were enabled to transcribe music accurately.",
        "author": "Adria Galin",
        "doi": "10.13140/RG.2.2.31541.52960",
        "keywords": "Automatic piano transcription, real-time systems, computer vision, pitch detection, music information retrieval",
        "title": "Design, Implementation and Validation of a Novel Real-Time Automatic Piano Transcription System",
        "type": "PHDTHESIS",
        "url": "https://www.researchgate.net/publication/334093755_Design_Implementation_and_Validation_of_a_Novel_Real-Time_Automatic_Piano_Transcription_System",
        "year": "2017"
    },
    "Maman2022": {
        "abstract": "Multi-instrument Automatic Music Transcription (AMT), or the decoding of a musical recording into semantic musical content, is one of the holy grails of Music Information Retrieval. Current AMT approaches are restricted to piano and (some) guitar recordings, due to difficult data collection. In order to overcome data collection barriers, previous AMT approaches attempt to employ musical scores in the form of a digitized version of the same song or piece. The scores are typically aligned using audio features and strenuous human intervention to generate training labels. We introduce NoteEM, a method for simultaneously training a transcriber and aligning the scores to their corresponding performances, in a fully-automated process. Using this unaligned supervision scheme, complemented by pseudo-labels and pitch-shift augmentation, our method enables training on in-the-wild recordings with unprecedented accuracy and instrumental variety. Using only synthetic data and unaligned supervision, we report SOTA note-level accuracy of the MAPS dataset, and large favorable margins on cross-dataset evaluations. We also demonstrate robustness and ease of use; we report comparable results when training on a small, easily obtainable, self-collected dataset, and we propose alternative labeling to the MusicNet dataset, which we show to be more accurate.",
        "author": "Ben Maman and Amit H. Bermano",
        "doi": "10.48550/arXiv.2204.13668",
        "keywords": "Music information retrieval, Automatic music transcription, Real-world data, Semi-supervised learning",
        "title": "Unaligned supervision for automatic music transcription in the wild",
        "type": "ARTICLE",
        "url": "https://arxiv.org/abs/2204.13668",
        "year": "2022"
    },
    "Puri2017": {
        "abstract": "In this paper, the literature survey of the automatic music transcription system have been presented. Now a day's most of the research work going on Music transcription and it is considered to be a most difficult problem even by human experts and current music transcription systems fail to match human performance. As compare to Monophonic AMT the Polyphonic AMT is a difficult problem because in polyphonic concurrently sounding notes from one or more instruments cause a complex interaction and overlap of harmonics in the acoustic signal. So we concentrate on all methods of polyphonic AMT. Most of the music transcription systems were developed for the instruments typically used in western music like Piano, Guitar etc. but very less paper/work has been publishing in the domain of harmonium note transcription which is widely used the instrument in Indian musical concerts.",
        "author": "Surekha B. Puri and S. P. Mahajan",
        "doi": "10.1109/ICCUBEA.2017.8463892",
        "keywords": "Automatic Music Transcription, HMM, LPC, KNN, monophonic, Music Language Models, polyphonic, PLCA, RNN, SVM",
        "title": "Review on Automatic Music Transcription System",
        "type": "INPROCEEDINGS",
        "url": "https://ieeexplore.ieee.org/document/8463892",
        "year": "2017"
    },
    "Roman2020": {
        "abstract": "This work presents an end-to-end method based on deep neural networks for audio-to-score music transcription of monophonic excerpts. Unlike existing music transcription methods, which normally perform pitch estimation, the proposed approach is formulated as an end-to-end task that outputs a notation-level music score. Using an audio file as input, modeled as a sequence of frames, a deep neural network is trained to provide a sequence of music symbols encoding a score, including key and time signatures, barlines, notes (with their pitch spelling and duration) and rests. Our framework is based on a Convolutional Recurrent Neural Network (CRNN) with Connectionist Temporal Classification (CTC) loss function trained in an end-to-end fashion, without requiring to align the input frames with the output symbols. A total of 246,870 incipits from the R\u00e9pertoire International des Sources Musicales online catalog were synthesized using different timbres and tempos to build the training data. Alternative input representations (raw audio, Short-Time Fourier Transform (STFT), log-spaced STFT and Constant-Q transform) were evaluated for this task, as well as different output representations (Plaine & Easie Code, Kern, and a purpose-designed output). Results show that it is feasible to directly infer score representations from audio files and most errors come from music notation ambiguities and metering (time signatures and barlines).",
        "author": "Miguel A. Rom\u00e1n, Antonio Pertusa and Jorge Calvo-Zaragoza",
        "doi": "10.1016/j.eswa.2020.113769",
        "keywords": "Automatic music transcription, Audio processing, Neural networks, Audio to score, Monophonic music",
        "title": "Data representations for audio-to-score monophonic music transcription",
        "type": "ARTICLE",
        "url": "https://www.sciencedirect.com/science/article/pii/S0957417420305935",
        "year": "2020"
    },
    "Skoki2019": {
        "abstract": "Sopela is a traditional hand-made woodwind instrument, commonly played in pair, characteristic to the Istrian peninsula in western Croatia. Its piercing sound, accompanied by two-part singing in the hexatonic Istrian scale, is registered in the UNESCO Representative List of the Intangible Cultural Heritage of Humanity. This paper presents an insight study of automatic music transcription (AMT) for sopele tunes. The process of converting audio inputs into human-readable musical scores involves multi-pitch detection and note tracking. The proposed solution supports this process by utilising frequency-feature extraction, supervised machine learning (ML) algorithms, and postprocessing heuristics. We determined the most favourable tone-predicting model by applying grid search for two state-of-the-art ML techniques, optionally coupled with frequency-feature extraction. The model achieved promising transcription accuracy for both monophonic and polyphonic music sources encompassed in the originally developed dataset. In addition, we developed a proof-of-concept AMT system, comprised of a client mobile application and a server-side API. While the mobile application records, tags and uploads audio sources, the back-end server applies the presented procedure for converting recorded music into a common notation to be delivered as a transcription result. We thus demonstrate how collecting and preserving traditional sopele music, performed in real-life occasions, can be effortlessly accomplished on-the-go.",
        "author": "Arian Skoki, Sandi Ljubic, Jonatan Lerga and Ivan \u0160tajduhar",
        "doi": "https://doi.org/10.1016/j.patrec.2019.09.024",
        "keywords": "Automatic music transcription, Traditional woodwind instrument, Sopele, Discrete Fourier transform, Machine learning",
        "title": "Automatic music transcription for traditional woodwind instruments sopele",
        "type": "ARTICLE",
        "url": "https://www.sciencedirect.com/science/article/pii/S0167865519302703",
        "year": "2019"
    },
    "Vaca2019": {
        "abstract": "A real-time automatic music transcription (AMT) system has a great potential for applications and interactions between people and music, such as the popular devices Amazon Echo and Google Home. This paper thus presents a design on chord recognition with the Zync7000 Field-Programmable Gate Array (FPGA), capable of sampling analog frequency signals through a microphone and, in real time, showing sheet music on a smart phone app that corresponds to the user's playing. We demonstrate the design of audio sampling on programming logic and the implementation of frequency transform and vector building on programming system, which is an embedded ARM core on the Zync FPGA. Experimental results show that the logic design spends 574 slices of look-up-tables (LUTs) and 792 slices of flip-flops. Due to the dynamic power consumption on programming system (1399 mW) being significantly higher than the dynamic power dissipation on programming logic (7 mW), the future work of this platform is to design intelligent property (IP) for algorithms of frequency transform, pitch class profile (PCP), and pattern matching with hardware description language (HDL), making the entire system-on-chip (SoC) able to be taped out as an application-specific design for consumer electronics.",
        "author": "Kevin Vaca, Archit Gajjar and Xiaokun Yang",
        "doi": "10.1109/ISVLSI.2019.00075",
        "keywords": "Automatic music transcription (AMT), Field-programmable gate array (FPGA), Pitch class profile (PCP), System-on-chip (SoC)",
        "title": "Real-Time Automatic Music Transcription (AMT) with Zync FPGA",
        "type": "INPROCEEDINGS",
        "url": "https://ieeexplore.ieee.org/document/8839533",
        "year": "2019"
    },
    "Wu2020": {
        "abstract": "Multi-instrument automatic music transcription (AMT) is a critical but less investigated problem in the field of music information retrieval (MIR). With all the difficulties faced by traditional AMT research, multi-instrument AMT needs further investigation on high-level music semantic modeling, efficient training methods for multiple attributes, and a clear problem scenario for system performance evaluation. In this article, we propose a multi-instrument AMT method, with signal processing techniques specifying pitch saliency, novel deep learning techniques, and concepts partly inspired by multi-object recognition, instance segmentation, and image-to-image translation in computer vision. The proposed method is flexible for all the sub-tasks in multi-instrument AMT, including multi-instrument note tracking, a task that has rarely been investigated before. State-of-the-art performance is also reported in the sub-task of multi-pitch streaming.",
        "author": "Yu-Te Wu, Berlin Chen and Li Su",
        "doi": "10.1109/TASLP.2020.3030482",
        "keywords": "Automatic music transcription, Deep learning, Multi-pitch estimation, Multi-pitch streaming, Self-attention",
        "title": "Multi-Instrument Automatic Music Transcription With Self-Attention-Based Instance Segmentation",
        "type": "ARTICLE",
        "url": "https://ieeexplore.ieee.org/document/9222310",
        "year": "2020"
    }
}});